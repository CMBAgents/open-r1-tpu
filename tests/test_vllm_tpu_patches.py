"""Cover the build-time patches applied to the vendored vLLM TPU service."""

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
PATCHES_DIR = REPO_ROOT / "docker/vllm-tpu/patches"


def load_patch(name: str):
    spec = importlib.util.spec_from_file_location(name, PATCHES_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lazy_text_config():
    return load_patch("lazy_text_config_fallback")


def test_dockerfile_runs_every_committed_patch():
    dockerfile = (REPO_ROOT / "docker/vllm-tpu/Dockerfile").read_text()

    assert list(PATCHES_DIR.glob("*.py"))
    assert "COPY patches /tmp/patches" in dockerfile
    assert 'for patch in /tmp/patches/*.py; do python "$patch"; done' in dockerfile


def test_rewrites_the_wrapped_upstream_call(lazy_text_config):
    source = (
        "        self.hidden_size = getattr(model_config.hf_config, 'hidden_size',\n"
        "                                   model_config.hf_config"
        ".text_config.hidden_size)\n"
    )

    patched, count = lazy_text_config.patch_source(source)

    assert count == 1
    assert patched == (
        "        self.hidden_size = "
        "model_config.hf_config.get_text_config().hidden_size\n"
    )
    compile(f"class Model:\n    def __init__(self):\n{patched}", "<patched>", "exec")


def test_rewrites_every_shape_lookup_regardless_of_layout(lazy_text_config):
    source = (
        'heads = getattr(cfg, "num_attention_heads", '
        "cfg.text_config.num_attention_heads)\n"
        "layers = getattr(\n"
        "    cfg,\n"
        "    'num_hidden_layers',\n"
        "    cfg.text_config.num_hidden_layers,\n"
        ")\n"
    )

    patched, count = lazy_text_config.patch_source(source)

    assert count == 2
    assert patched == (
        "heads = cfg.get_text_config().num_attention_heads\n"
        "layers = cfg.get_text_config().num_hidden_layers\n"
    )
    compile(patched, "<patched>", "exec")


def test_leaves_unrelated_getattr_defaults_alone(lazy_text_config):
    source = (
        "a = getattr(cfg, 'hidden_size', 4096)\n"
        "b = getattr(cfg, 'hidden_size', other.text_config.hidden_size)\n"
        "c = getattr(cfg, 'hidden_size', cfg.text_config.head_dim)\n"
        "d = cfg.text_config.hidden_size\n"
    )

    patched, count = lazy_text_config.patch_source(source)

    assert count == 0
    assert patched == source


def test_reports_the_environment_it_cannot_patch(lazy_text_config, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit, match="tpu_inference is not installed"):
        lazy_text_config.package_root()
