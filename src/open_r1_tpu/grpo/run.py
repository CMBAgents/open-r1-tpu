"""GRPO reinforcement learning on top of a merged SFT export, via Tunix.

Run with::

  python -m open_r1_tpu.grpo.run --config \
    recipes/OpenR1-Distill-Qwen2.5-Math-1.5B/grpo/config_grpo.yaml

See ``GRPO.md`` at the repository root for prerequisites (the merged SFT
export must exist and be staged), staging steps, and a preflight checklist --
this module performs no preflight of its own, matching
``sft.run``/``sft.preflight``'s split.

Design, verified directly against this project's pinned Tunix commit
(``google-tunix @ 984bf89b...`` in ``pyproject.toml``) rather than assumed
from an example notebook, since Tunix's RL API has moved under the ``rl_``
prefix since some published examples were written (``GRPOLearner``'s
constructor parameter is ``rl_engine``, not the ``rl_cluster`` an older
example passes; ``RLCluster`` is confirmed to still be an alias for
``RLEngine``, so passing an ``RLCluster`` instance as ``rl_engine=`` is
correct). See ``tunix.readthedocs.io``'s rollout page for why the "vanilla"
rollout engine (in-process JAX/Flax, no external server) is the right choice
for a single-chip v6e-1: it is what that page recommends for single-device
setups, and it is what every published Tunix GRPO example uses.

The actor and the reference are the *same* merged SFT checkpoint, loaded
twice: the actor gets a fresh LoRA adapter (the only thing GRPO trains here),
the reference stays the frozen full model that GRPO's KL term constrains
policy updates against (README's "Checkpoints and GRPO handoff" section).
Both loads go through ``model.loading.create_model``, the same helper
``sft.run.run`` uses for SFT, so a change to how this project loads a local
safetensors checkpoint cannot drift between the two training stages.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
from typing import Any

from open_r1_tpu.core.config import load_config
from open_r1_tpu.core.logging import LOG_LEVELS, configure_logging
from open_r1_tpu.grpo.data import load_grpo_prompts
from open_r1_tpu.grpo.rewards import DEFAULT_REWARD_FNS
from open_r1_tpu.model.export import export_model
from open_r1_tpu.model.loading import absolute_checkpoint_dir, create_model
from open_r1_tpu.model.metrics import metrics_logger_options
from open_r1_tpu.model.optimizer import create_optimizer
from open_r1_tpu.model.tokenizing import assistant_turn_end_id

LOGGER = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML GRPO recipe path")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=sorted(LOG_LEVELS),
        type=str.lower,
        help="Stderr log level. debug restores the demoted library logs.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Tunix-style overrides such as grpo.num_generations=4",
    )
    return parser.parse_args()


def validate_grpo_config(config: dict[str, Any]) -> None:
    """Fail early for a GRPO recipe mistake, before any TPU time is spent.

    A different shape from ``core.config.validate_config`` (SFT: a packed,
    tokenized dataset, no rollout section; GRPO: raw prompts, a rollout
    section, a required LoRA actor), so it is its own validator rather than
    an extension of that one -- mirroring how
    ``evaluation.run.validate_eval_config`` is its own function rather than
    a variant of the training validator.
    """
    for section in (
        "model",
        "tokenizer",
        "dataset",
        "optimizer",
        "training",
        "grpo",
        "rollout",
    ):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    mesh = config["model"].get("mesh", {})
    shape = mesh.get("shape")
    axis_names = mesh.get("axis_names")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(size, int) and size > 0 for size in shape)
    ):
        raise ValueError("model.mesh.shape must be a non-empty list of integers")
    if not isinstance(axis_names, list) or len(axis_names) != len(shape):
        raise ValueError("model.mesh.axis_names must have one name per mesh dimension")

    if not config["model"].get("lora_config"):
        raise ValueError(
            "model.lora_config is required: the GRPO actor trains a LoRA "
            "adapter over a frozen reference, never a full fine-tune here"
        )

    batch_size = config["dataset"].get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("dataset.batch_size must be a positive integer")

    max_steps = config["training"].get("max_steps")
    if not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("training.max_steps must be a positive integer")
    if not isinstance(config["training"].get("checkpoint_dir"), str):
        raise ValueError("training.checkpoint_dir must be a string path")

    rollout = config["rollout"]
    for key in ("max_prompt_length", "max_tokens_to_generate", "kv_cache_size"):
        value = rollout.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"rollout.{key} must be a positive integer")
    # tunix.readthedocs.io/en/latest/rollout.html: the vanilla rollout raises
    # ValueError at run time if this does not hold. Checked here too so the
    # failure names the recipe field instead of surfacing after model
    # loading has already spent several minutes of TPU time.
    if (
        rollout["kv_cache_size"]
        < rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    ):
        raise ValueError(
            "rollout.kv_cache_size must be at least "
            "max_prompt_length + max_tokens_to_generate"
        )

    grpo = config["grpo"]
    for key in ("num_generations", "num_iterations"):
        value = grpo.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"grpo.{key} must be a positive integer")
    for key in ("beta", "epsilon"):
        value = grpo.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"grpo.{key} must be a number")

    export = config.get("export", {})
    if export.get("enabled") and not export.get("i_have_verified_qwen2_lora_export"):
        raise ValueError(
            "export.enabled requires export.i_have_verified_qwen2_lora_export: "
            "true, set only after confirming on the VM that "
            "tunix.models.automodel.get_model_module(model_name, "
            "ModelModule.PARAMS) exposes save_lora_merged_model_as_safetensors "
            "for this model family. README's 'Checkpoints and GRPO handoff' "
            "section: Tunix's merged-LoRA exporter is confirmed for Qwen3 "
            "only, and this recipe's model is Qwen2. Until then, the "
            "Tunix/Orbax LoRA checkpoint under training.checkpoint_dir "
            "(actor/<step>/model_params) is the durable artifact; see "
            "GRPO.md."
        )


def run(config: dict[str, Any]) -> None:
    import jax
    from tunix.cli.utils import model as model_utils
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
    from tunix.rl.rollout import base_rollout
    from tunix.sft import checkpoint_options, metrics_logger
    from tunix.sft import utils as sft_utils
    from tunix.utils import mesh as mesh_utils

    mesh_shape = tuple(config["model"]["mesh"]["shape"])
    axis_names = tuple(config["model"]["mesh"]["axis_names"])
    if math.prod(mesh_shape) != jax.device_count():
        raise ValueError(
            f"Configured mesh {mesh_shape} needs {math.prod(mesh_shape)} devices, "
            f"but JAX sees {jax.device_count()}. Override model.mesh.shape."
        )
    mesh = mesh_utils.create_mesh(mesh_shape, axis_names)

    # Reference: the frozen merged SFT export, no LoRA -- GRPO's KL anchor.
    reference_config = copy.deepcopy(config)
    reference_config["model"].pop("lora_config", None)
    reference_model, tokenizer_path = create_model(reference_config, mesh)

    # Actor: the identical weights, with a fresh LoRA adapter to train.
    actor_model, _ = create_model(config, mesh)
    if not sft_utils.is_lora_enabled(actor_model):
        raise RuntimeError(
            "LoRA was requested but Tunix found no matching modules. Check "
            "model.lora_config.module_path before training."
        )

    tokenizer = model_utils.create_tokenizer(config["tokenizer"], tokenizer_path)
    if config["tokenizer"].get("chat_template"):
        tokenizer.tokenizer.chat_template = config["tokenizer"]["chat_template"]
    eos_token_id = assistant_turn_end_id(tokenizer)

    train_ds, eval_ds = load_grpo_prompts(config["dataset"], tokenizer)

    training = config["training"]
    max_steps = int(training["max_steps"])
    checkpointing = checkpoint_options.checkpointing_options_from_dict(
        training.get("checkpointing_options", {})
    )
    metrics = metrics_logger_options(config, metrics_logger)

    rollout = config["rollout"]
    cluster_config = rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=bool(training.get("offload_to_cpu", False)),
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=create_optimizer(config, max_steps),
            eval_every_n_steps=int(training.get("eval_every_n_steps", 50)),
            max_steps=max_steps,
            mini_batch_size=int(
                training.get("mini_batch_size", config["dataset"]["batch_size"])
            ),
            train_micro_batch_size=int(config["dataset"]["batch_size"]),
            metrics_logging_options=metrics,
            checkpoint_root_directory=absolute_checkpoint_dir(
                training["checkpoint_dir"]
            ),
            checkpointing_options=checkpointing,
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=int(rollout["max_tokens_to_generate"]),
            max_prompt_length=int(rollout["max_prompt_length"]),
            kv_cache_size=int(rollout["kv_cache_size"]),
            temperature=float(rollout.get("temperature", 0.9)),
            top_p=rollout.get("top_p"),
            top_k=rollout.get("top_k"),
            eos_tokens=[eos_token_id],
        ),
    )

    grpo = config["grpo"]
    grpo_config = GRPOConfig(
        num_generations=int(grpo["num_generations"]),
        num_iterations=int(grpo.get("num_iterations", 1)),
        beta=float(grpo.get("beta", 0.04)),
        epsilon=float(grpo.get("epsilon", 0.2)),
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    grpo_trainer = GRPOLearner(
        rl_engine=rl_cluster,
        algo_config=grpo_config,
        reward_fns=list(DEFAULT_REWARD_FNS),
    )

    LOGGER.info(
        "Starting GRPO: model=%s mesh=%s max_steps=%d num_generations=%d beta=%s",
        config["model"]["model_id"],
        mesh_shape,
        max_steps,
        grpo_config.num_generations,
        grpo_config.beta,
    )
    # Same reason as sft.run.run: Tunix 0.1.8's PeftTrainer (which
    # GRPOLearner's actor update goes through) still reads JAX's legacy
    # thread-local physical mesh, so jax.set_mesh(mesh) alone is not enough.
    with mesh:
        grpo_trainer.train(train_ds, eval_ds)

    if config["model"].get("model_source") == "huggingface":
        local_model_path = config["model"].get("model_download_path")
    else:
        local_model_path = config["model"].get("model_path")
    if not local_model_path:
        raise ValueError("No local base-model path is available for merged export")
    export_model(
        config=config,
        model=actor_model,
        tokenizer=tokenizer,
        local_model_path=local_model_path,
    )


def main() -> None:
    args = _parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
    config = load_config(args.config, args.overrides, validator=validate_grpo_config)
    run(config)


if __name__ == "__main__":
    main()
