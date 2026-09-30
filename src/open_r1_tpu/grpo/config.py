"""GRPO recipe validation."""

from __future__ import annotations

from typing import Any

from open_r1_tpu.core.config import check_sections, reject_unknown_keys
from open_r1_tpu.grpo.rewards import reward_fns_from_names
from open_r1_tpu.model.export import EXPORT_KEYS
from open_r1_tpu.model.loading import validate_mesh_config
from open_r1_tpu.model.metrics import METRICS_KEYS, validate_wandb_config
from open_r1_tpu.model.optimizer import OPTIMIZER_KEYS

SECTIONS = (
    "model",
    "tokenizer",
    "dataset",
    "optimizer",
    "training",
    "grpo",
    "rollout",
)
OPTIONAL_SECTIONS = ("export",)

# The keys of each section this project defines, as grpo.data and grpo.run
# read them. `model`, `tokenizer` and `training.checkpointing_options` pass
# through to Tunix and are not checked.
DATASET_KEYS = frozenset(
    {
        "name",
        "config",
        "train_split",
        "data_files",
        "question_column",
        "answer_column",
        "system_prompt_file",
        "batch_size",
        "eval_batch_size",
        "max_prompt_length",
        "num_train_epochs",
        "max_examples",
        "eval_fraction",
        "eval_max_examples",
        "seed",
    }
)
TRAINING_KEYS = METRICS_KEYS | {
    "max_steps",
    "eval_every_n_steps",
    "eval_rollouts_path",
    "checkpoint_dir",
    "checkpointing_options",
    "offload_to_cpu",
    "mini_batch_size",
    "train_micro_batch_size",
    "rollout_micro_batch_size",
    "compute_logps_micro_batch_size",
}
GRPO_KEYS = frozenset(
    {
        "reward_functions",
        "num_generations",
        "num_iterations",
        "beta",
        "epsilon",
        "loss_agg_mode",
        "kl_loss_mode",
    }
)
ROLLOUT_KEYS = frozenset(
    {
        "max_prompt_length",
        "max_tokens_to_generate",
        "kv_cache_size",
        "temperature",
        "top_p",
        "top_k",
        "eos_token_ids",
    }
)

# GRPOConfig values the pinned Tunix accepts (tunix/rl/common.py's
# aggregate_loss and compute_kl_divergence). Unset, Tunix defaults to
# sequence-mean-token-mean and the plain "kl" estimator.
LOSS_AGG_MODES = frozenset(
    {
        "token-mean",
        "sequence-mean-token-mean",
        "sequence-mean-token-scale",
        "seq-mean-token-sum",
        "sequence-mean-token-sum-norm",
    }
)
KL_LOSS_MODES = frozenset({"kl", "mse_kl", "low_var_kl"})


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def validate_grpo_config(config: dict[str, Any]) -> None:
    """Fail early for GRPO recipe mistakes that would otherwise waste TPU time."""
    check_sections(config, SECTIONS, OPTIONAL_SECTIONS)
    dataset, training = config["dataset"], config["training"]
    rollout, grpo = config["rollout"], config["grpo"]
    reject_unknown_keys("dataset", dataset, DATASET_KEYS)
    reject_unknown_keys("optimizer", config["optimizer"], OPTIMIZER_KEYS)
    reject_unknown_keys("training", training, TRAINING_KEYS)
    reject_unknown_keys("grpo", grpo, GRPO_KEYS)
    reject_unknown_keys("rollout", rollout, ROLLOUT_KEYS)
    reject_unknown_keys("export", config.get("export", {}), EXPORT_KEYS)
    validate_mesh_config(config["model"])

    if not config["model"].get("lora_config"):
        raise ValueError(
            "model.lora_config is required: the GRPO actor trains a LoRA "
            "adapter over a frozen reference, never a full fine-tune here"
        )

    batch_size = dataset.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("dataset.batch_size must be a positive integer")

    max_steps = training.get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("training.max_steps must be a positive integer")
    if not isinstance(training.get("checkpoint_dir"), str):
        raise ValueError("training.checkpoint_dir must be a string path")
    validate_wandb_config(training)
    eval_rollouts_path = training.get("eval_rollouts_path")
    if eval_rollouts_path is not None and (
        not isinstance(eval_rollouts_path, str) or not eval_rollouts_path
    ):
        raise ValueError("training.eval_rollouts_path must be a non-empty string")
    if eval_rollouts_path and not float(dataset.get("eval_fraction", 0.0)):
        raise ValueError(
            "training.eval_rollouts_path needs dataset.eval_fraction > 0: "
            "there are no eval rollouts to record otherwise"
        )

    eval_batch_size = dataset.get("eval_batch_size")
    if eval_batch_size is not None and not _is_positive_int(eval_batch_size):
        raise ValueError("dataset.eval_batch_size must be a positive integer")

    for key in (
        "mini_batch_size",
        "train_micro_batch_size",
        "rollout_micro_batch_size",
        "compute_logps_micro_batch_size",
    ):
        value = training.get(key)
        if value is not None and not _is_positive_int(value):
            raise ValueError(f"training.{key} must be a positive integer")
    mini_batch = training.get("mini_batch_size", batch_size)
    train_micro = training.get("train_micro_batch_size", batch_size)
    if mini_batch % train_micro:
        raise ValueError(
            "training.train_micro_batch_size must divide training.mini_batch_size "
            "(dataset.batch_size by default)"
        )

    for key in ("max_prompt_length", "max_tokens_to_generate", "kv_cache_size"):
        value = rollout.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"rollout.{key} must be a positive integer")
    # The vanilla rollout raises on this too, but only after model loading.
    if (
        rollout["kv_cache_size"]
        < rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    ):
        raise ValueError(
            "rollout.kv_cache_size must be at least "
            "max_prompt_length + max_tokens_to_generate"
        )

    eos_token_ids = rollout.get("eos_token_ids")
    if eos_token_ids is not None and (
        not isinstance(eos_token_ids, list)
        or not eos_token_ids
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in eos_token_ids
        )
    ):
        raise ValueError(
            "rollout.eos_token_ids must be a non-empty list of non-negative integers"
        )

    reward_names = grpo.get("reward_functions")
    if reward_names is not None:
        if (
            not isinstance(reward_names, list)
            or not reward_names
            or any(not isinstance(name, str) for name in reward_names)
        ):
            raise ValueError("grpo.reward_functions must be a non-empty list of names")
        reward_fns_from_names(reward_names)  # raises on an unknown name
    for key in ("num_generations", "num_iterations"):
        value = grpo.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"grpo.{key} must be a positive integer")
    for key in ("beta", "epsilon"):
        value = grpo.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"grpo.{key} must be a number")
    for key, allowed in (
        ("loss_agg_mode", LOSS_AGG_MODES),
        ("kl_loss_mode", KL_LOSS_MODES),
    ):
        value = grpo.get(key)
        if value is not None and value not in allowed:
            raise ValueError(f"grpo.{key} must be one of {sorted(allowed)}")
