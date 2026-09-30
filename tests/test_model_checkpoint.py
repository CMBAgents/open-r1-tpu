"""`open_r1_tpu.model.checkpoint`, without a TPU: JAX and Tunix are faked."""

import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from open_r1_tpu.model import checkpoint

DISTILL_RECIPE = (
    Path(__file__).parents[1] / "recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml"
)


def test_inference_config_carries_lora_only_when_asked():
    assert "lora_config" not in checkpoint.inference_model_config(
        "/models/base", 0, model_name="qwen3-1.7b-base", mesh_shape=(1, 1)
    )
    lora = {"rank": 8, "alpha": 8.0, "module_path": ".*q_proj"}
    assert (
        checkpoint.inference_model_config(
            "/models/base",
            0,
            model_name="qwen3-1.7b-base",
            mesh_shape=(1, 1),
            lora_config=lora,
        )["lora_config"]
        == lora
    )


def test_inference_config_loads_weights_in_float32_without_splash_attention():
    # Under the pinned Tunix the jitted bfloat16 Qwen2 forward returns all-NaN
    # logits for long KV caches, and splash attention would let the sampler's
    # left padding reach real tokens.
    config = checkpoint.inference_model_config(
        "/models/base", 0, model_name="qwen2.5-math-1.5b", mesh_shape=(2, 2)
    )

    assert config["dtype"] == "float32"
    assert config["load_dtype"] == "float32"
    assert config["use_flash_attention"] is False


def test_inference_config_carries_the_discovered_mesh_shape():
    assert checkpoint.inference_model_config(
        "/models/base", 0, model_name="qwen3-1.7b-base", mesh_shape=(1, 4)
    )["mesh"] == {"shape": [1, 4], "axis_names": ["fsdp", "tp"]}


def test_model_settings_are_derived_from_the_local_config(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"_name_or_path":"Qwen/Qwen2.5-Math-1.5B","num_key_value_heads":2}',
        encoding="utf-8",
    )

    assert checkpoint.model_settings_for_path(str(tmp_path)) == (
        "qwen2.5-math-1.5b",
        2,
        None,
    )


def test_model_settings_accept_an_explicit_name_when_source_metadata_is_missing(
    tmp_path,
):
    (tmp_path / "config.json").write_text('{"num_key_value_heads":2}', encoding="utf-8")

    assert checkpoint.model_settings_for_path(str(tmp_path), "qwen2.5-math-1.5b") == (
        "qwen2.5-math-1.5b",
        2,
        None,
    )


def test_a_re_based_rope_theta_is_read_from_the_export_and_served(tmp_path):
    # Tunix takes rope_theta from its registered config for the model name, so
    # a base re-based to a different theta would otherwise be served at the
    # stock value and answer badly with nothing to say why.
    (tmp_path / "config.json").write_text(
        '{"_name_or_path":"Qwen/Qwen2.5-Math-1.5B","num_key_value_heads":2,'
        '"rope_theta":300000}',
        encoding="utf-8",
    )

    assert checkpoint.model_settings_for_path(str(tmp_path)) == (
        "qwen2.5-math-1.5b",
        2,
        300000.0,
    )
    served = checkpoint.inference_model_config(
        str(tmp_path),
        0,
        model_name="qwen2.5-math-1.5b",
        mesh_shape=(1, 1),
        rope_theta=300000.0,
    )
    assert served["rope_theta"] == 300000.0
    # A config that does not name one must not invent a value.
    assert "rope_theta" not in checkpoint.inference_model_config(
        str(tmp_path), 0, model_name="qwen2.5-math-1.5b", mesh_shape=(1, 1)
    )


def test_resolve_model_dir_requires_safetensors(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        checkpoint.resolve_model_dir(str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match=r"No model\.safetensors"):
        checkpoint.resolve_model_dir(str(tmp_path))
    (tmp_path / "model.safetensors").touch()
    assert checkpoint.resolve_model_dir(str(tmp_path)) == str(tmp_path.resolve())


def test_mesh_shape_uses_every_visible_tpu_device():
    devices = [SimpleNamespace(platform="tpu", id=device_id) for device_id in range(4)]

    assert checkpoint.mesh_shape_for_devices(devices, num_kv_heads=8) == (1, 4)


def test_qwen2_5_mesh_uses_fully_sharded_data_parallelism_after_tp():
    devices = [SimpleNamespace(platform="tpu", id=device_id) for device_id in range(4)]

    assert checkpoint.mesh_shape_for_devices(devices, num_kv_heads=2) == (2, 2)
    assert checkpoint.pad_input_strings_for_fsdp(["prompt"], fsdp_size=2) == [
        "prompt",
        "prompt",
    ]


def test_mesh_shape_rejects_no_devices_or_non_tpu_devices():
    with pytest.raises(RuntimeError, match="at least one visible TPU"):
        checkpoint.mesh_shape_for_devices([], num_kv_heads=8)
    with pytest.raises(RuntimeError, match="every visible JAX device"):
        checkpoint.mesh_shape_for_devices(
            [SimpleNamespace(platform="cpu", id=0)], num_kv_heads=8
        )


def test_mesh_shape_rejects_a_tpu_count_that_cannot_tensor_parallelize_qwen3():
    devices = [SimpleNamespace(platform="tpu", id=device_id) for device_id in range(3)]

    with pytest.raises(RuntimeError, match="dividing its 8 KV heads"):
        checkpoint.mesh_shape_for_devices(devices, num_kv_heads=8)


def test_tunix_mesh_context_enters_the_mesh_and_silences_its_deprecation():
    entered = []

    class Mesh:
        def __enter__(self):
            warnings.warn(
                "`with mesh:` context manager has been deprecated.",
                DeprecationWarning,
                stacklevel=2,
            )
            entered.append(True)

        def __exit__(self, *exc_info):
            return False

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with checkpoint.tunix_mesh_context(Mesh()):
            assert entered == [True]


def _fake_runtime(monkeypatch, captured: dict[str, Any]) -> SimpleNamespace:
    """Stand-ins for JAX, Tunix and create_model, recording what they receive."""
    runtime = SimpleNamespace(
        mesh=object(),
        tokenizer=object(),
        sampler=object(),
        model=SimpleNamespace(
            config=SimpleNamespace(num_layers=28, num_kv_heads=2, head_dim=128)
        ),
    )

    def create_mesh(shape, axis_names):
        captured["mesh"] = (shape, axis_names)
        return runtime.mesh

    def create_model(config, received_mesh):
        captured["model"] = (config, received_mesh)
        return runtime.model, "tokenizer-path"

    def cache_config(**kwargs):
        captured["cache"] = kwargs
        return SimpleNamespace(**kwargs)

    def sampler(**kwargs):
        captured["sampler"] = kwargs
        return runtime.sampler

    fake_jax: Any = ModuleType("jax")
    fake_jax.devices = lambda: [
        SimpleNamespace(platform="tpu", id=device_id) for device_id in range(4)
    ]
    fake_model_utils: Any = ModuleType("tunix.cli.utils.model")
    fake_model_utils.create_tokenizer = lambda _config, _path: runtime.tokenizer
    fake_sampler_lib: Any = ModuleType("tunix.generate.sampler")
    fake_sampler_lib.CacheConfig = cache_config
    fake_sampler_lib.Sampler = sampler
    fake_mesh_utils: Any = ModuleType("tunix.utils.mesh")
    fake_mesh_utils.create_mesh = create_mesh
    fake_model_loading: Any = ModuleType("open_r1_tpu.model.loading")
    fake_model_loading.create_model = create_model
    monkeypatch.setattr(
        checkpoint,
        "model_settings_for_path",
        lambda *_args: ("qwen2.5-math-1.5b", 2, 300000.0),
    )

    for name, module in {
        "jax": fake_jax,
        "tunix": ModuleType("tunix"),
        "tunix.cli": ModuleType("tunix.cli"),
        "tunix.cli.utils": ModuleType("tunix.cli.utils"),
        "tunix.cli.utils.model": fake_model_utils,
        "tunix.generate": ModuleType("tunix.generate"),
        "tunix.generate.sampler": fake_sampler_lib,
        "tunix.utils": ModuleType("tunix.utils"),
        "tunix.utils.mesh": fake_mesh_utils,
        "open_r1_tpu.model.loading": fake_model_loading,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return runtime


def test_load_sampler_wires_qwen2_5_into_the_four_chip_mesh(monkeypatch):
    captured: dict[str, Any] = {}
    fake = _fake_runtime(monkeypatch, captured)

    loaded = checkpoint.load_sampler("/models/base", seed=42, cache_size=1152)

    assert loaded == (fake.mesh, fake.tokenizer, fake.sampler, 2)
    assert captured["mesh"] == ((2, 2), ("fsdp", "tp"))
    assert captured["model"] == (
        {
            "model": checkpoint.inference_model_config(
                "/models/base",
                42,
                model_name="qwen2.5-math-1.5b",
                mesh_shape=(2, 2),
                rope_theta=300000.0,
            ),
            "tokenizer": checkpoint.tokenizer_config("/models/base"),
        },
        fake.mesh,
    )
    assert captured["cache"]["cache_size"] == 1152
    assert captured["sampler"]["transformer"] is fake.model


def test_load_sampler_restores_a_lora_recipe_into_its_own_geometry(monkeypatch):
    captured: dict[str, Any] = {}
    fake = _fake_runtime(monkeypatch, captured)
    lora = {"rank": 64, "alpha": 64.0, "module_path": ".*q_proj"}
    monkeypatch.setattr(
        checkpoint,
        "recipe_restore_settings",
        lambda _recipe: (lora, "artifacts/run/checkpoints"),
    )

    def restore_checkpoint(model, checkpoint_dir, step, *, lora_only):
        captured["restore"] = (model, checkpoint_dir, step, lora_only)
        return 1500

    monkeypatch.setattr(checkpoint, "restore_checkpoint", restore_checkpoint)

    checkpoint.load_sampler(
        "/models/base",
        seed=42,
        cache_size=1152,
        recipe="lora.yaml",
        checkpoint_dir="elsewhere/checkpoints",
        step=1500,
    )

    assert captured["model"][0]["model"]["lora_config"] == lora
    assert captured["restore"] == (fake.model, "elsewhere/checkpoints", 1500, True)


def test_recipe_restore_settings_match_a_lora_recipe(tmp_path):
    recipe = tmp_path / "lora.yaml"
    recipe.write_text(
        f"extends: {DISTILL_RECIPE}\n"
        "model:\n"
        "  lora_config: {module_path: '.*q_proj', rank: 64, alpha: 64.0}\n"
        "training:\n"
        "  checkpoint_dir: artifacts/lora-run/checkpoints\n"
    )
    lora_config, checkpoint_dir = checkpoint.recipe_restore_settings(str(recipe))

    # Restoring under a different geometry than training wrote would produce
    # confident nonsense rather than an error, so these must come from there.
    assert lora_config is not None
    assert lora_config["rank"] == 64
    assert lora_config["alpha"] == 64.0
    assert checkpoint_dir == "artifacts/lora-run/checkpoints"


def test_full_finetune_recipe_restores_all_parameters():
    # The distill recipe trains all parameters, so its checkpoints carry the
    # full model state and restore must not be limited to LoRA adapters.
    lora_config, checkpoint_dir = checkpoint.recipe_restore_settings(
        str(DISTILL_RECIPE)
    )

    assert lora_config is None
    assert checkpoint_dir.endswith("OpenR1-Distill-Qwen2.5-Math-1.5B/checkpoints")


def _checkpoint_root(tmp_path, steps):
    for step in steps:
        (tmp_path / str(step)).mkdir()
    (tmp_path / "not-a-step").mkdir()
    return str(tmp_path)


def test_available_steps_ignores_non_step_directories(tmp_path):
    root = _checkpoint_root(tmp_path, [1500, 1000])

    assert checkpoint.available_steps(root) == [1000, 1500]


def test_available_steps_tolerates_a_missing_root(tmp_path):
    assert checkpoint.available_steps(str(tmp_path / "absent")) == []


def test_resolve_step_defaults_to_the_latest_written():
    assert checkpoint.resolve_step("/absent", None) is None


def test_resolve_step_rejects_a_step_that_was_never_saved(tmp_path):
    # A run stopped at step 1744 last saved 1500, and 1744 is the step its log
    # reported.
    root = _checkpoint_root(tmp_path, [1000, 1500])

    with pytest.raises(FileNotFoundError, match="1000, 1500"):
        checkpoint.resolve_step(root, 1744)


def test_resolve_step_accepts_a_step_that_was_saved(tmp_path):
    root = _checkpoint_root(tmp_path, [1000, 1500])

    assert checkpoint.resolve_step(root, 1000) == 1000


def test_importing_the_module_does_not_import_jax_or_tunix():
    # The scripts parse their arguments before anything initialises a TPU.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, open_r1_tpu.model.checkpoint; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('jax', 'tunix')))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    assert completed.stdout.strip() == "[]"
