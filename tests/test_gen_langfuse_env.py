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
