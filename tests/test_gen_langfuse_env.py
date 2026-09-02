"""`scripts/gen_langfuse_env.sh` writes docker/langfuse/.env and
configs/tracing.yaml with every cross-referenced value already consistent.

The script needs neither Docker nor a TPU -- it shells out to `openssl` and
writes two text files under a REPO_ROOT it derives from its own path -- so
these run from a laptop against a copied `scripts/` directory in a tmp tree.
"""

from __future__ import annotations

import re
import shutil
import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "gen_langfuse_env.sh"
COMPOSE_PATH = REPO_ROOT / "docker" / "langfuse" / "docker-compose.yaml"


# Every ${VAR} / ${VAR:-default} the compose file interpolates, from its
# non-comment lines (the header prose says "${VAR:-default}" literally). The
# generated .env must define all of them, so no service silently falls back to
# a placeholder default.
def _compose_vars() -> set[str]:
    found: set[str] = set()
    for line in COMPOSE_PATH.read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        code = line.split(" #", 1)[0]
        found.update(re.findall(r"\$\{([A-Z0-9_]+)", code))
    return found


COMPOSE_VARS = _compose_vars()


def _run(args, cwd):
    return subprocess.run(
        ["bash", *args],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
        cwd=cwd,
    )


def _tree(tmp_path):
    """A tmp REPO_ROOT holding just a copy of the script."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    copy = scripts_dir / SCRIPT_PATH.name
    shutil.copy(SCRIPT_PATH, copy)
    copy.chmod(0o755)
    return copy


def _parse_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key] = value
    return out


def test_the_script_is_syntactically_valid():
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_generates_both_files_mode_600(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script)], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr

    env_file = tmp_path / "docker" / "langfuse" / ".env"
    tracing_file = tmp_path / "configs" / "tracing.yaml"
    assert env_file.is_file()
    assert tracing_file.is_file()
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert stat.S_IMODE(tracing_file.stat().st_mode) == 0o600


def test_env_defines_every_variable_the_compose_file_reads(tmp_path):
    script = _tree(tmp_path)
    assert _run([str(script)], cwd=tmp_path).returncode == 0
    env = _parse_env(tmp_path / "docker" / "langfuse" / ".env")
    missing = COMPOSE_VARS - env.keys()
    assert not missing, f"generated .env is missing compose vars: {sorted(missing)}"


def test_cross_referenced_values_are_consistent(tmp_path):
    script = _tree(tmp_path)
    assert _run([str(script)], cwd=tmp_path).returncode == 0
    env = _parse_env(tmp_path / "docker" / "langfuse" / ".env")

    # DATABASE_URL embeds POSTGRES_PASSWORD verbatim (no .env interpolation).
    assert env["DATABASE_URL"] == (
        f"postgresql://postgres:{env['POSTGRES_PASSWORD']}@postgres:5432/postgres"
    )

    # Langfuse's S3 client credentials are the MinIO root credentials, for both
    # the event and the media bucket.
    for prefix in ("LANGFUSE_S3_EVENT_UPLOAD", "LANGFUSE_S3_MEDIA_UPLOAD"):
        assert env[f"{prefix}_ACCESS_KEY_ID"] == env["MINIO_ROOT_USER"]
        assert env[f"{prefix}_SECRET_ACCESS_KEY"] == env["MINIO_ROOT_PASSWORD"]

    # The auth URL and the tracing config name the same port as the web port.
    assert env["NEXTAUTH_URL"].endswith(f":{env['LANGFUSE_WEB_PORT']}")
    tracing = (tmp_path / "configs" / "tracing.yaml").read_text()
    assert f"port: {env['LANGFUSE_WEB_PORT']}" in tracing

    # With no flags this is a single-host deployment: the interface the server
    # publishes on and the address the client dials are both loopback.
    assert env["LANGFUSE_WEB_BIND"] == "127.0.0.1"
    assert "host: 127.0.0.1" in tracing


def test_secrets_are_freshly_generated_not_placeholders(tmp_path):
    script = _tree(tmp_path)
    assert _run([str(script)], cwd=tmp_path).returncode == 0
    env = _parse_env(tmp_path / "docker" / "langfuse" / ".env")

    assert re.fullmatch(r"[0-9a-f]{48}", env["POSTGRES_PASSWORD"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["SALT"])
    assert re.fullmatch(r"[0-9a-f]{64}", env["ENCRYPTION_KEY"])
    assert re.fullmatch(r"pk-lf-[0-9a-f-]{36}", env["LANGFUSE_INIT_PROJECT_PUBLIC_KEY"])
    assert re.fullmatch(r"sk-lf-[0-9a-f-]{36}", env["LANGFUSE_INIT_PROJECT_SECRET_KEY"])
    assert "changeme" not in Path(tmp_path / "docker" / "langfuse" / ".env").read_text()


def test_refuses_to_overwrite_without_force(tmp_path):
    script = _tree(tmp_path)
    assert _run([str(script)], cwd=tmp_path).returncode == 0
    env_file = tmp_path / "docker" / "langfuse" / ".env"
    first = env_file.read_text()

    again = _run([str(script)], cwd=tmp_path)
    assert again.returncode == 1
    assert "Refusing to overwrite" in again.stderr
    assert env_file.read_text() == first  # untouched

    forced = _run([str(script), "--force"], cwd=tmp_path)
    assert forced.returncode == 0, forced.stderr
    assert env_file.read_text() != first  # secrets rotated


def test_no_tracing_config_flag_writes_only_the_env(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script), "--no-tracing-config"], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "docker" / "langfuse" / ".env").is_file()
    assert not (tmp_path / "configs" / "tracing.yaml").exists()


def test_print_keys_emits_export_lines_matching_the_env(tmp_path):
    script = _tree(tmp_path)
    assert _run([str(script)], cwd=tmp_path).returncode == 0
    env = _parse_env(tmp_path / "docker" / "langfuse" / ".env")

    completed = _run([str(script), "--print-keys"], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    # Leading blank line guards a target file with no trailing newline.
    assert lines[0] == ""
    assert lines[1] == (
        f"export LANGFUSE_PUBLIC_KEY={env['LANGFUSE_INIT_PROJECT_PUBLIC_KEY']}"
    )
    assert lines[2] == (
        f"export LANGFUSE_SECRET_KEY={env['LANGFUSE_INIT_PROJECT_SECRET_KEY']}"
    )


def test_print_keys_without_an_env_file_errors(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script), "--print-keys"], cwd=tmp_path)
    assert completed.returncode == 1
    assert "gen_langfuse_env.sh" in completed.stderr


# --- Two hosts -------------------------------------------------------------
# The stack and the evaluation need not share a machine, and then .env lives on
# one and configs/tracing.yaml on the other. `--web-bind` is the server's side
# of that one endpoint and `--langfuse-host` the client's; neither has a
# non-loopback default, because both are deployment values that must not be
# committed (AGENTS.md, tests/test_tracing_no_hardcoded_values.py).

REMOTE = "192.0.2.10"  # TEST-NET-1, RFC 5737: never a real deployment.


def test_web_bind_sets_the_published_interface(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script), "--web-bind", REMOTE], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    env = _parse_env(tmp_path / "docker" / "langfuse" / ".env")
    assert env["LANGFUSE_WEB_BIND"] == REMOTE


def test_nextauth_url_stays_localhost_when_web_bind_does_not(tmp_path):
    # NEXTAUTH_URL is the origin a *browser* sees, and the UI is still reached
    # through an SSH port-forward onto localhost. Rewriting it to track the
    # bind address would break sign-in for the supported way of viewing it.
    script = _tree(tmp_path)
    assert _run([str(script), "--web-bind", REMOTE], cwd=tmp_path).returncode == 0
    env = _parse_env(tmp_path / "docker" / "langfuse" / ".env")
    assert env["NEXTAUTH_URL"] == f"http://localhost:{env['LANGFUSE_WEB_PORT']}"


def test_non_loopback_web_bind_warns_about_reachability(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script), "--web-bind", REMOTE], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    # The two things that become the operator's problem: who may reach the
    # port, and that the UI forward no longer targets loopback.
    assert REMOTE in completed.stderr
    assert "evaluation host only" in completed.stderr
    assert "-L" in completed.stderr


def test_the_loopback_default_does_not_warn(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script)], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert "not loopback" not in completed.stderr


def test_langfuse_host_sets_the_address_the_client_dials(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script), "--langfuse-host", REMOTE], cwd=tmp_path)
    assert completed.returncode == 0, completed.stderr
    assert f"host: {REMOTE}" in (tmp_path / "configs" / "tracing.yaml").read_text()


def test_langfuse_port_moves_both_halves_together(tmp_path):
    script = _tree(tmp_path)
    assert _run([str(script), "--langfuse-port", "3100"], cwd=tmp_path).returncode == 0
    env = _parse_env(tmp_path / "docker" / "langfuse" / ".env")
    tracing = (tmp_path / "configs" / "tracing.yaml").read_text()
    assert env["LANGFUSE_WEB_PORT"] == "3100"
    assert env["NEXTAUTH_URL"].endswith(":3100")
    assert "port: 3100" in tracing


def test_tracing_only_writes_the_client_half_and_no_secrets(tmp_path):
    # The evaluation host needs the endpoint, not key material: nothing should
    # generate a .env on a machine that has no stack to run.
    script = _tree(tmp_path)
    completed = _run(
        [str(script), "--tracing-only", "--langfuse-host", REMOTE], cwd=tmp_path
    )
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "docker").exists()
    tracing = tmp_path / "configs" / "tracing.yaml"
    assert f"host: {REMOTE}" in tracing.read_text()
    assert stat.S_IMODE(tracing.stat().st_mode) == 0o600


def test_tracing_only_requires_a_host(tmp_path):
    # Defaulting to loopback here would write a config pointing at a machine
    # with no Langfuse on it, and the failure would surface much later.
    script = _tree(tmp_path)
    completed = _run([str(script), "--tracing-only"], cwd=tmp_path)
    assert completed.returncode == 2
    assert "--langfuse-host" in completed.stderr
    assert not (tmp_path / "configs").exists()


def test_tracing_only_conflicts_with_no_tracing_config(tmp_path):
    script = _tree(tmp_path)
    completed = _run(
        [
            str(script),
            "--tracing-only",
            "--no-tracing-config",
            "--langfuse-host",
            REMOTE,
        ],
        cwd=tmp_path,
    )
    assert completed.returncode == 2
    assert "would write nothing" in completed.stderr


def test_tracing_only_refuses_to_overwrite_without_force(tmp_path):
    script = _tree(tmp_path)
    args = [str(script), "--tracing-only", "--langfuse-host", REMOTE]
    assert _run(args, cwd=tmp_path).returncode == 0
    again = _run(args, cwd=tmp_path)
    assert again.returncode == 1
    assert "Refusing to overwrite" in again.stderr
    assert _run([*args, "--force"], cwd=tmp_path).returncode == 0


def test_an_address_flag_rejects_a_url(tmp_path):
    # Both files want a bare host; the client composes scheme and port itself.
    # A URL here boots a stack the client then cannot reach, with nothing
    # obviously wrong in either file.
    script = _tree(tmp_path)
    for flag in ("--web-bind", "--langfuse-host"):
        completed = _run([str(script), flag, f"http://{REMOTE}:3000"], cwd=tmp_path)
        assert completed.returncode == 2, flag
        assert "bare host" in completed.stderr
        assert not (tmp_path / "docker").exists()


def test_an_address_flag_rejects_a_host_port_pair(tmp_path):
    script = _tree(tmp_path)
    completed = _run([str(script), "--web-bind", f"{REMOTE}:3000"], cwd=tmp_path)
    assert completed.returncode == 2
    assert "bare host" in completed.stderr


def test_langfuse_port_rejects_a_non_port(tmp_path):
    script = _tree(tmp_path)
    for bad in ("0", "70000", "web"):
        completed = _run([str(script), "--langfuse-port", bad], cwd=tmp_path)
        assert completed.returncode == 2, bad
        assert "TCP port number" in completed.stderr


def test_the_two_host_split_agrees_on_one_endpoint(tmp_path):
    """The pair of invocations README.md prescribes, run against two trees:
    what the server publishes is what the client dials, and each host writes
    only its own half."""
    server = tmp_path / "server"
    client = tmp_path / "client"
    server.mkdir()
    client.mkdir()
    server_script = _tree(server)
    client_script = _tree(client)

    assert (
        _run(
            [str(server_script), "--web-bind", REMOTE, "--no-tracing-config"],
            cwd=server,
        ).returncode
        == 0
    )
    assert (
        _run(
            [str(client_script), "--tracing-only", "--langfuse-host", REMOTE],
            cwd=client,
        ).returncode
        == 0
    )

    env = _parse_env(server / "docker" / "langfuse" / ".env")
    tracing = (client / "configs" / "tracing.yaml").read_text()
    assert f"host: {env['LANGFUSE_WEB_BIND']}" in tracing
    assert f"port: {env['LANGFUSE_WEB_PORT']}" in tracing
    assert not (server / "configs").exists()
    assert not (client / "docker").exists()
