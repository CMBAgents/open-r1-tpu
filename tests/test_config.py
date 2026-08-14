from pathlib import Path

import pytest

from open_r1_tpu.config import load_config, parse_override
from open_r1_tpu.sft import (
    _SteppedTrainingMetricsBackend,
    _metrics_logger_options,
    _wandb_backend_kwargs,
)


RECIPE = (
    Path(__file__).parents[1]
    / "recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml"
)


def test_parse_override_uses_yaml_types():
    assert parse_override("training.max_steps=12") == ("training.max_steps", 12)
    assert parse_override("export.enabled=false") == ("export.enabled", False)
    assert parse_override("model.mesh.shape=[1, 8]") == (
        "model.mesh.shape",
        [1, 8],
    )


def test_load_config_applies_nested_overrides():
    config = load_config(
        RECIPE,
        ["dataset.max_examples=128", "model.mesh.shape=[1, 8]"],
    )
    assert config["dataset"]["max_examples"] == 128
    assert config["model"]["mesh"]["shape"] == [1, 8]


def test_default_recipe_targets_one_32gb_tpu():
    config = load_config(RECIPE)

    assert config["model"]["mesh"]["shape"] == [1, 1]
    assert config["dataset"]["batch_size"] == 1
    assert config["dataset"]["max_length"] == 1024
    assert config["training"]["gradient_accumulation_steps"] == 8


def test_invalid_mesh_is_rejected():
    with pytest.raises(ValueError, match="axis_names"):
        load_config(RECIPE, ["model.mesh.axis_names=[fsdp]"])


def test_wandb_can_be_disabled_for_local_runs():
    config = load_config(RECIPE, ["training.wandb.enabled=false"])
    assert config["training"]["wandb"]["enabled"] is False
    assert _wandb_backend_kwargs(config) == {"mode": "disabled"}


def test_wandb_backend_receives_run_metadata_and_resolved_config():
    config = load_config(RECIPE, ["training.wandb.entity=my-team"])
    kwargs = _wandb_backend_kwargs(config)

    assert kwargs["entity"] == "my-team"
    assert kwargs["group"] == "qwen3-1.7b-reasoning-distillation"
    assert kwargs["job_type"] == "sft"
    assert kwargs["tags"] == ["tpu", "sft", "lora", "qwen3"]
    assert kwargs["dir"] == config["training"]["metrics_log_dir"]
    assert kwargs["config"] is config


def test_wandb_adapter_filters_unstepped_and_non_training_metrics():
    class RecordingBackend:
        def __init__(self):
            self.calls = []
            self.closed = False

        def log_scalar(self, event, value, **kwargs):
            self.calls.append((event, value, kwargs))

        def close(self):
            self.closed = True

    recording_backend = RecordingBackend()
    backend = _SteppedTrainingMetricsBackend(recording_backend)

    backend.log_scalar("/train/loss", 1.25, step=3)
    backend.log_scalar("/eval/loss", 1.5, step=3)
    backend.log_scalar("/jax/orbax/write/gbytes_per_sec", 10.0)
    backend.log_scalar("/jax/core/compile/backend_compile_duration", 12.0)
    backend.log_scalar("/train/perplexity", 3.5)
    backend.close()

    assert recording_backend.calls == [
        ("/train/loss", 1.25, {"step": 3}),
        ("/eval/loss", 1.5, {"step": 3}),
    ]
    assert recording_backend.closed is True


def test_metrics_options_use_custom_backends():
    class Options:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Backend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMetricsLogger:
        MetricsLoggerOptions = Options
        TensorboardBackend = Backend
        WandbBackend = Backend

    config = load_config(RECIPE)
    options = _metrics_logger_options(config, FakeMetricsLogger)
    factories = options.kwargs["backend_kwargs"]["custom_backend"]

    assert len(factories) == 2
    assert factories[0]().kwargs == {
        "log_dir": config["training"]["metrics_log_dir"],
        "flush_every_n_steps": config["training"]["flush_every_n_steps"],
    }
    assert isinstance(factories[1](), _SteppedTrainingMetricsBackend)

    disabled_config = load_config(RECIPE, ["training.wandb.enabled=false"])
    disabled_options = _metrics_logger_options(disabled_config, FakeMetricsLogger)
    disabled_factories = disabled_options.kwargs["backend_kwargs"][
        "custom_backend"
    ]
    assert len(disabled_factories) == 1


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("training.wandb.mode=invalid", "mode"),
        ("training.wandb.tags=[tpu, 1]", "tags"),
    ],
)
def test_invalid_wandb_config_is_rejected(override, error):
    with pytest.raises(ValueError, match=error):
        load_config(RECIPE, [override])
