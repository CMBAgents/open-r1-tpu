from pathlib import Path

import pytest

from open_r1_tpu.config import load_config, parse_override
from open_r1_tpu.sft import (
    _absolute_checkpoint_dir,
    _metrics_logger_options,
    _SteppedTrainingMetricsBackend,
    _wandb_backend_kwargs,
)

RECIPE = (
    Path(__file__).parents[1]
    / "recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml"
)
INSTRUCT_RECIPE = (
    Path(__file__).parents[1] / "recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml"
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
    assert config["training"]["project_name"] == "open-r1-tpu"
    assert config["training"]["wandb"]["entity"] is None


@pytest.mark.parametrize(
    "recipe", [RECIPE, INSTRUCT_RECIPE], ids=["distill", "instruct"]
)
def test_every_recipe_targets_one_device_and_names_no_entity(recipe):
    config = load_config(recipe)

    assert config["model"]["mesh"]["shape"] == [1, 1]
    # Deployment-specific values belong in the environment, not in the recipe.
    assert config["training"]["wandb"]["entity"] is None
    # The splash attention kernel requires its block size to divide the
    # sequence length, and fails at trace time rather than at load time.
    block_size = config["model"]["flash_attention_block_size"]
    assert config["dataset"]["max_length"] % block_size == 0


def test_instruct_recipe_does_not_filter_on_reasoning_tags():
    config = load_config(INSTRUCT_RECIPE)

    # SmolTalk carries no <think>/</think> traces. Requiring them would filter
    # every example and train on an empty dataset instead of raising.
    assert config["dataset"]["require_reasoning_tags"] is False
    # prepare_messages injects a system prompt wherever an example lacks one, so
    # a reasoning instruction here would precede every response that ignores it.
    assert config["dataset"]["system_prompt"] is None
    assert config["dataset"]["assistant_only_loss"] is True


def test_instruct_recipe_reads_staged_parquet_rather_than_the_hub():
    config = load_config(INSTRUCT_RECIPE)

    assert config["dataset"]["name"] == "parquet"
    data_files = config["dataset"]["data_files"]
    # data_files carries no split mapping, so every match lands in the train
    # split. Globbing the test shard alongside them would train on held-out data.
    assert "train-" in data_files
    assert "test" not in data_files


def test_instruct_recipe_packs_and_trains_without_eval():
    config = load_config(INSTRUCT_RECIPE)

    # Packing carries segment geometry the trainer needs; without it two-thirds
    # of every 2048-token window is padding on this corpus.
    assert config["dataset"]["packing"] is True
    # Eval is off: the measured 250-step eval spikes reached 99.9% of HBM, and
    # batch_size 2 spends that headroom on throughput instead.
    assert config["dataset"]["eval_fraction"] == 0.0
    assert config["dataset"]["batch_size"] == 2
    # Effective batch stays 32 windows per optimizer step.
    assert config["training"]["gradient_accumulation_steps"] == 16


def test_instruct_recipe_exports_where_the_distill_recipe_can_read_it():
    instruct = load_config(INSTRUCT_RECIPE)
    distill = load_config(RECIPE)

    # The stages are chained through the merged export, so they must not share a
    # checkpoint directory or the second run would restore the first one's state.
    assert instruct["export"]["enabled"] is True
    assert (
        instruct["training"]["checkpoint_dir"] != distill["training"]["checkpoint_dir"]
    )


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
    resolved = _absolute_checkpoint_dir("artifacts/run/checkpoints")
    assert Path(resolved).is_absolute()
    assert resolved.endswith("artifacts/run/checkpoints")


def test_absolute_and_remote_checkpoint_dirs_are_preserved():
    assert _absolute_checkpoint_dir("/data/run/checkpoints") == (
        "/data/run/checkpoints"
    )
    # Resolving a URI against the working directory would corrupt the scheme.
    assert _absolute_checkpoint_dir("gs://bucket/run/checkpoints") == (
        "gs://bucket/run/checkpoints"
    )


def test_default_recipe_checkpoint_dir_resolves_to_an_absolute_path():
    config = load_config(RECIPE)
    assert Path(
        _absolute_checkpoint_dir(config["training"]["checkpoint_dir"])
    ).is_absolute()


def test_default_recipe_has_a_bounded_eval_split():
    config = load_config(RECIPE)
    # Without a held-out split, train loss is the only signal there is.
    assert config["dataset"]["eval_fraction"] > 0.0
    assert config["dataset"]["eval_max_examples"] == 64


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
