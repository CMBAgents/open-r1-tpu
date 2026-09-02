"""`open_r1_tpu.tracing.config`: the one place tracing deployment values are
loaded and validated. See its module docstring for the schema.
"""

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import pytest
import yaml

from open_r1_tpu.tracing import config as tracing_config

EXAMPLE_CONFIG = Path(__file__).parents[1] / "configs" / "tracing.example.yaml"

VALID_CONFIG = {
    "langfuse": {"host": "127.0.0.1", "port": 3000},
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
    assert config["langfuse"]["host"]
    assert config["langfuse"]["port"] == 3000


def test_a_valid_config_loads(tmp_path):
    config = tracing_config.load_tracing_config(write_config(tmp_path))
    assert config["langfuse"]["host"] == "127.0.0.1"


def test_a_missing_section_names_it(tmp_path):
    path = write_config(tmp_path, {"langfuse": None})
    with pytest.raises(ValueError, match="Missing configuration section: langfuse"):
        tracing_config.load_tracing_config(path)


def test_a_missing_key_within_a_section_names_it(tmp_path):
    payload = copy.deepcopy(VALID_CONFIG)
    del payload["langfuse"]["host"]
    path = tmp_path / "tracing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"langfuse\.host"):
        tracing_config.load_tracing_config(path)


def test_an_unknown_key_suggests_the_near_match(tmp_path):
    path = write_config(tmp_path, {"langfuse": {"hosst": "typo"}})
    with pytest.raises(ValueError, match=r"langfuse\.hosst.*did you mean 'host'"):
        tracing_config.load_tracing_config(path)


def test_an_unknown_top_level_section_is_rejected(tmp_path):
    path = write_config(tmp_path, {"langfusee": {"host": "x"}})
    with pytest.raises(ValueError, match="Unknown configuration section 'langfusee'"):
        tracing_config.load_tracing_config(path)


def test_a_typoed_dotted_override_is_caught_after_merging(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ValueError, match=r"langfuse\.hosst"):
        tracing_config.load_tracing_config(path, ["langfuse.hosst=typo"])


def test_a_valid_dotted_override_applies(tmp_path):
    path = write_config(tmp_path)
    config = tracing_config.load_tracing_config(path, ["langfuse.port=5000"])
    assert config["langfuse"]["port"] == 5000


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("langfuse", "port", 0),
        ("langfuse", "port", 70000),
        ("langfuse", "port", "not-a-port"),
        ("langfuse", "host", ""),
    ],
)
def test_invalid_values_are_rejected(tmp_path, section, key, value):
    path = write_config(tmp_path, {section: {key: value}})
    with pytest.raises(ValueError):
        tracing_config.load_tracing_config(path)


# --- build_langfuse_client ----------------------------------------------------


def test_build_langfuse_client_uses_the_langfuse_section(tmp_path, monkeypatch):
    captured = {}

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("langfuse")
    # setattr, not `fake_module.Langfuse = ...`: ModuleType has no such
    # attribute to assign to, which pyright rejects outright.
    setattr(fake_module, "Langfuse", _FakeLangfuse)  # noqa: B010
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    config = tracing_config.load_tracing_config(write_config(tmp_path))
    tracing_config.build_langfuse_client(config)
    assert captured["base_url"] == "http://127.0.0.1:3000"
