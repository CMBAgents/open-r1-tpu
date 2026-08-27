"""`open_r1_tpu.tracing.config`: the one place tracing deployment values are
loaded and validated. See its module docstring for the schema.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from open_r1_tpu.tracing import config as tracing_config

EXAMPLE_CONFIG = Path(__file__).parents[1] / "configs" / "tracing.example.yaml"

VALID_CONFIG = {
    "gcs": {"bucket": "a-bucket", "prefix_template": "traces/{recipe}/{timestamp}"},
    "proxy": {
        "port": 4000,
        "upstream_base_url": "http://127.0.0.1:8000/v1",
        "image": "ghcr.io/example/litellm:v1@sha256:" + "a" * 64,
    },
    "langfuse": {"host": "127.0.0.1", "port": 3000},
    "ingester": {"poll_secs": 30, "state_dir": "artifacts/tracing/ingest-state"},
}


def write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = copy.deepcopy(VALID_CONFIG)
    for section, keys in (overrides or {}).items():
        if keys is None:
            payload.pop(section, None)
            continue
        payload.setdefault(section, {}).update(keys)
    path = tmp_path / "tracing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# --- loading and validation --------------------------------------------------


def test_the_committed_example_config_loads():
    config = tracing_config.load_tracing_config(EXAMPLE_CONFIG)
    assert config["gcs"]["bucket"]
    assert config["proxy"]["port"] == 4000


def test_a_valid_config_loads(tmp_path):
    config = tracing_config.load_tracing_config(write_config(tmp_path))
    assert config["langfuse"]["host"] == "127.0.0.1"


def test_a_missing_section_names_it(tmp_path):
    path = write_config(tmp_path, {"ingester": None})
    with pytest.raises(ValueError, match="Missing configuration section: ingester"):
        tracing_config.load_tracing_config(path)


def test_a_missing_key_within_a_section_names_it(tmp_path):
    payload = copy.deepcopy(VALID_CONFIG)
    del payload["gcs"]["bucket"]
    path = tmp_path / "tracing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"gcs\.bucket"):
        tracing_config.load_tracing_config(path)


def test_an_unknown_key_suggests_the_near_match(tmp_path):
    path = write_config(tmp_path, {"gcs": {"buckit": "typo"}})
    with pytest.raises(ValueError, match=r"gcs\.buckit.*did you mean 'bucket'"):
        tracing_config.load_tracing_config(path)


def test_an_unknown_top_level_section_is_rejected(tmp_path):
    path = write_config(tmp_path, {"proxyy": {"port": 1}})
    with pytest.raises(ValueError, match="Unknown configuration section 'proxyy'"):
        tracing_config.load_tracing_config(path)


def test_a_typoed_dotted_override_is_caught_after_merging(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ValueError, match=r"gcs\.buckit"):
        tracing_config.load_tracing_config(path, ["gcs.buckit=typo"])


def test_a_valid_dotted_override_applies(tmp_path):
    path = write_config(tmp_path)
    config = tracing_config.load_tracing_config(path, ["proxy.port=5000"])
    assert config["proxy"]["port"] == 5000


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("proxy", "port", 0),
        ("proxy", "port", 70000),
        ("langfuse", "port", "not-a-port"),
        ("ingester", "poll_secs", 0),
        ("ingester", "poll_secs", -1),
        ("gcs", "bucket", ""),
        ("gcs", "prefix_template", "traces/{oops}"),
    ],
)
def test_invalid_values_are_rejected(tmp_path, section, key, value):
    path = write_config(tmp_path, {section: {key: value}})
    with pytest.raises(ValueError):
        tracing_config.load_tracing_config(path)


def test_proxy_image_requires_an_immutable_digest(tmp_path):
    path = write_config(
        tmp_path, {"proxy": {"image": "ghcr.io/example/litellm:latest"}}
    )
    with pytest.raises(ValueError, match="sha256"):
        tracing_config.load_tracing_config(path)


# --- prefix rendering ---------------------------------------------------------


def test_render_prefix_is_deterministic():
    template = "traces/{recipe}/{timestamp}"
    first = tracing_config.render_prefix(template, recipe="tier0", timestamp="t1")
    second = tracing_config.render_prefix(template, recipe="tier0", timestamp="t1")
    assert first == second == "traces/tier0/t1"


def test_render_prefix_varies_with_its_inputs():
    template = "traces/{recipe}/{timestamp}"
    a = tracing_config.render_prefix(template, recipe="tier0", timestamp="t1")
    b = tracing_config.render_prefix(template, recipe="tier1", timestamp="t1")
    assert a != b


# --- --export-env --------------------------------------------------------


def test_export_env_lines_round_trip_through_a_shell_eval(tmp_path):
    # A value with quotes and whitespace, to stress shlex.quote rather than a
    # value that would round-trip through eval even unquoted.
    path = write_config(tmp_path, {"gcs": {"bucket": 'a bucket\'s "name"'}})
    config = tracing_config.load_tracing_config(path)
    lines = tracing_config.export_env_lines(config)
    script = "\n".join(lines) + '\necho "$TRACE_GCS_BUCKET"\n'

    completed = subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=True
    )

    assert completed.stdout.rstrip("\n") == 'a bucket\'s "name"'


def test_export_env_includes_rendered_prefix_only_with_recipe_and_timestamp(tmp_path):
    config = tracing_config.load_tracing_config(write_config(tmp_path))

    without = tracing_config.export_env_lines(config)
    assert not any("TRACE_GCS_PREFIX=" in line for line in without)

    with_both = tracing_config.export_env_lines(config, recipe="tier0", timestamp="t1")
    assert any(line.startswith("export TRACE_GCS_PREFIX=") for line in with_both)


def test_export_env_requires_recipe_and_timestamp_together(tmp_path):
    config = tracing_config.load_tracing_config(write_config(tmp_path))
    with pytest.raises(ValueError, match="recipe and timestamp"):
        tracing_config.export_env_lines(config, recipe="tier0")


def test_export_env_cli_prints_export_lines(tmp_path):
    path = write_config(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "open_r1_tpu.tracing.config",
            "--config",
            str(path),
            "--export-env",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).parents[1],
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
    )
    assert "export TRACE_PROXY_PORT=4000" in completed.stdout


def test_cli_without_export_env_errors():
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "open_r1_tpu.tracing.config",
                "--config",
                str(EXAMPLE_CONFIG),
            ],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parents[1],
            env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
        )
