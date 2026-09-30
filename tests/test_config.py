from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config, parse_override
from open_r1_tpu.model.loading import absolute_checkpoint_dir
from open_r1_tpu.model.metrics import (
    SteppedTrainingMetricsBackend,
    metrics_logger_options,
    wandb_backend_kwargs,
)

RECIPE = (
    Path(__file__).parents[1] / "recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml"
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


def test_distill_recipe_matches_its_tested_setup():
    config = load_config(RECIPE)

    # Four chips as [fsdp, tp] = [2, 2]: tp halves the float32 logits of a
    # 26,624-token window, which do not fit one chip whole.
    assert config["model"]["mesh"]["shape"] == [2, 2]
    # LoRA freezes the tied embedding matrix, whose rows carry <|im_end|>.
    assert "lora_config" not in config["model"]
    # The config-edited RoPE-300k base, staged locally so its config.json is
    # what the export copies.
    assert config["model"]["model_source"] == "local"
    assert config["model"]["model_path"].endswith("Qwen2.5-Math-1.5B-RoPE-300k")
    # The splash kernel needs its block size to divide the sequence length,
    # and fails at trace time rather than at load time.
    block_size = config["model"]["flash_attention_block_size"]
    assert config["dataset"]["max_length"] % block_size == 0
    # Deployment-specific values belong in the environment, not the recipe.
    assert config["training"]["wandb"]["entity"] is None
    assert config["training"]["project_name"] == "open-r1-tpu"


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("dataset.overlength_policy=clip", "overlength_policy"),
        ("dataset.message_schema=3", "message_schema"),
        ("dataset.message_schema={from: human}", "unknown"),
        ("dataset.message_schema={role_map: {human: 1}}", "role_map"),
    ],
)
def test_invalid_dataset_policy_config_is_rejected(override, error):
    with pytest.raises(ValueError, match=error):
        load_config(RECIPE, [override])


def test_invalid_mesh_is_rejected():
    with pytest.raises(ValueError, match="axis_names"):
        load_config(RECIPE, ["model.mesh.axis_names=[fsdp]"])


def test_wandb_can_be_disabled_for_local_runs():
    config = load_config(RECIPE, ["training.wandb.enabled=false"])
    assert config["training"]["wandb"]["enabled"] is False
    assert wandb_backend_kwargs(config) == {"mode": "disabled"}


def test_wandb_backend_receives_run_metadata_and_resolved_config():
    config = load_config(RECIPE, ["training.wandb.entity=my-team"])
    kwargs = wandb_backend_kwargs(config)

    assert kwargs["entity"] == "my-team"
    assert kwargs["group"] == "qwen2.5-math-1.5b-reasoning-distillation"
    assert kwargs["job_type"] == "sft"
    assert kwargs["tags"] == config["training"]["wandb"]["tags"]
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
    backend = SteppedTrainingMetricsBackend(recording_backend)

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
    options = metrics_logger_options(config, FakeMetricsLogger)
    factories = options.kwargs["backend_kwargs"]["custom_backend"]

    assert len(factories) == 2
    assert factories[0]().kwargs == {
        "log_dir": config["training"]["metrics_log_dir"],
        "flush_every_n_steps": config["training"]["flush_every_n_steps"],
    }
    assert isinstance(factories[1](), SteppedTrainingMetricsBackend)

    disabled_config = load_config(RECIPE, ["training.wandb.enabled=false"])
    disabled_options = metrics_logger_options(disabled_config, FakeMetricsLogger)
    disabled_factories = disabled_options.kwargs["backend_kwargs"]["custom_backend"]
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


def test_relative_checkpoint_dir_is_made_absolute():
    # Orbax raises "Checkpoint path should be absolute" for a relative
    # directory, which the recipe uses by default.
    resolved = absolute_checkpoint_dir("artifacts/run/checkpoints")
    assert Path(resolved).is_absolute()
    assert resolved.endswith("artifacts/run/checkpoints")


def test_absolute_and_remote_checkpoint_dirs_are_preserved():
    assert absolute_checkpoint_dir("/data/run/checkpoints") == ("/data/run/checkpoints")
    # Resolving a URI against the working directory would corrupt the scheme.
    assert absolute_checkpoint_dir("gs://bucket/run/checkpoints") == (
        "gs://bucket/run/checkpoints"
    )


def test_recipe_checkpoint_dir_resolves_to_an_absolute_path():
    config = load_config(RECIPE)
    assert Path(
        absolute_checkpoint_dir(config["training"]["checkpoint_dir"])
    ).is_absolute()


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("dataset.eval_fraction=1.0", "eval_fraction"),
        ("dataset.eval_fraction=-0.1", "eval_fraction"),
        ("dataset.eval_max_examples=0", "eval_max_examples"),
        ("training.transcripts.enabled=maybe", "enabled"),
        ("training.transcripts.every_n_steps=0", "every_n_steps"),
        ("training.transcripts.max_new_tokens=-1", "max_new_tokens"),
        ("training.transcripts.prompts=[]", "prompts"),
        ("training.transcripts.prompts=[1, 2]", "prompts"),
    ],
)
def test_invalid_inspection_config_is_rejected(override, error):
    with pytest.raises(ValueError, match=error):
        load_config(RECIPE, [override])
