"""SFT recipe validation."""

from __future__ import annotations

from typing import Any

from open_r1_tpu.core.config import check_sections, reject_unknown_keys
from open_r1_tpu.model.export import EXPORT_KEYS
from open_r1_tpu.model.loading import validate_mesh_config
from open_r1_tpu.model.metrics import METRICS_KEYS, validate_wandb_config
from open_r1_tpu.model.optimizer import OPTIMIZER_KEYS
from open_r1_tpu.sft.data import OVERLENGTH_POLICIES, message_schema_from_config

SECTIONS = ("model", "tokenizer", "dataset", "optimizer", "training")
OPTIONAL_SECTIONS = ("export",)

# The keys of each section this project defines, as sft.data, sft.run and
# sft.transcripts read them. `model`, `tokenizer` and
# `training.checkpointing_options` pass through to Tunix and are not checked.
DATASET_KEYS = frozenset(
    {
        "name",
        "config",
        "train_split",
        "data_files",
        "messages_column",
        "message_schema",
        "system_prompt_file",
        "batch_size",
        "max_length",
        "overlength_policy",
        "packing",
        "num_train_epochs",
        "max_examples",
        "eval_fraction",
        "eval_max_examples",
        "seed",
        "assistant_only_loss",
        "require_reasoning_tags",
        "reasoning_start",
        "reasoning_end",
    }
)
TRAINING_KEYS = METRICS_KEYS | {
    "max_steps",
    "gradient_accumulation_steps",
    "eval_every_n_steps",
    "checkpoint_dir",
    "checkpointing_options",
    "data_sharding_axis",
    "max_inflight_computations",
    "transcripts",
}
TRANSCRIPTS_KEYS = frozenset(
    {
        "enabled",
        "every_n_steps",
        "prompts",
        "max_new_tokens",
        "max_prompt_length",
        "cache_size",
        "temperature",
        "seed",
        "output_path",
        "log_to_wandb",
    }
)


def validate_sft_config(config: dict[str, Any]) -> None:
    """Fail early for SFT recipe mistakes that would otherwise waste TPU time."""
    check_sections(config, SECTIONS, OPTIONAL_SECTIONS)
    dataset, training = config["dataset"], config["training"]
    reject_unknown_keys("dataset", dataset, DATASET_KEYS)
    reject_unknown_keys("optimizer", config["optimizer"], OPTIMIZER_KEYS)
    reject_unknown_keys("training", training, TRAINING_KEYS)
    reject_unknown_keys("export", config.get("export", {}), EXPORT_KEYS)
    validate_mesh_config(config["model"])

    batch_size = dataset.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("dataset.batch_size must be a positive integer")

    max_length = dataset.get("max_length")
    if not isinstance(max_length, int) or max_length < 2:
        raise ValueError("dataset.max_length must be at least 2")

    accumulation = training.get("gradient_accumulation_steps", 1)
    if not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError(
            "training.gradient_accumulation_steps must be a positive integer"
        )

    validate_wandb_config(training)

    # A policy or schema mismatch filters every example, so training would
    # run on an empty dataset rather than fail.
    if dataset.get("overlength_policy", "drop") not in OVERLENGTH_POLICIES:
        raise ValueError(
            "dataset.overlength_policy must be one of " + ", ".join(OVERLENGTH_POLICIES)
        )
    message_schema_from_config(dataset.get("message_schema"))

    eval_fraction = dataset.get("eval_fraction", 0.0)
    if not isinstance(eval_fraction, (int, float)) or not 0.0 <= eval_fraction < 1.0:
        raise ValueError("dataset.eval_fraction must be in [0.0, 1.0)")
    eval_max_examples = dataset.get("eval_max_examples")
    if eval_max_examples is not None and (
        not isinstance(eval_max_examples, int) or eval_max_examples <= 0
    ):
        raise ValueError("dataset.eval_max_examples must be a positive integer or null")

    transcripts = training.get("transcripts", {})
    if not isinstance(transcripts, dict):
        raise ValueError("training.transcripts must be a configuration mapping")
    reject_unknown_keys("training.transcripts", transcripts, TRANSCRIPTS_KEYS)
    if not isinstance(transcripts.get("enabled", False), bool):
        raise ValueError("training.transcripts.enabled must be a boolean")
    for key in ("every_n_steps", "max_new_tokens"):
        value = transcripts.get(key, 1)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.transcripts.{key} must be a positive integer")
    prompts = transcripts.get("prompts")
    if prompts is not None and (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) for prompt in prompts)
    ):
        raise ValueError(
            "training.transcripts.prompts must be a non-empty list of strings or null"
        )
