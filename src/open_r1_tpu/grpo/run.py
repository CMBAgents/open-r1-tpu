"""GRPO reinforcement learning with Tunix.

Run with::

  python -m open_r1_tpu.grpo.run --config \
    recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml

The starting model and the prompt corpus must already be staged on local
disk. The actor and the reference are the same checkpoint loaded twice: the
actor gets a fresh LoRA adapter, the only thing GRPO trains, and the reference
stays frozen as the anchor of GRPO's KL term. Rollouts run in process on
Tunix's "vanilla" engine, the one it recommends for a single host.

Generation stops on the first token of the chat template's assistant turn-end
sequence unless ``rollout.eos_token_ids`` lists the tokens to stop on. With
``dataset.eval_fraction`` set, Tunix scores held-out prompts every
``training.eval_every_n_steps`` but discards the completions;
``training.eval_rollouts_path`` appends each one, with its prompt, gold answer
and per-function rewards, to a JSONL file.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from collections.abc import Callable, Sequence
from typing import Any

from open_r1_tpu.core.cli import parse_recipe_args, recipe_parser
from open_r1_tpu.core.config import load_config
from open_r1_tpu.grpo.behaviour import build_behaviour_metric_fn
from open_r1_tpu.grpo.config import validate_grpo_config
from open_r1_tpu.grpo.data import load_grpo_prompts
from open_r1_tpu.grpo.rewards import reward_fns_from_names
from open_r1_tpu.model.export import export_model, local_base_model_path
from open_r1_tpu.model.loading import (
    absolute_checkpoint_dir,
    create_mesh,
    create_model,
    create_tokenizer,
    require_lora,
)
from open_r1_tpu.model.metrics import metrics_logger_options
from open_r1_tpu.model.optimizer import create_optimizer
from open_r1_tpu.model.tokenizing import assistant_turn_end_id

LOGGER = logging.getLogger(__name__)


def _is_per_completion_column(value: Any, width: int) -> bool:
    """True when ``value`` holds one entry per completion.

    Tunix passes extra dataset columns as numpy arrays, not lists, so this
    accepts any sized non-string of the right length; a narrower test would
    silently drop the columns the reward functions need.
    """
    if isinstance(value, (str, bytes, dict)):
        return False
    try:
        return len(value) == width
    except TypeError:
        return False


def build_eval_table_logger(
    max_chars: int = 4000,
) -> tuple[Callable[..., None], Callable[[], None]]:
    """Return ``(add, flush)`` that log each eval's rollouts as a W&B table.

    Tunix scores eval rollouts one batch at a time, so ``add`` collects rows
    until a call arrives for a different step, then logs the finished step as
    ``eval/rollouts`` against the ``global_step`` axis. ``flush`` logs what is
    pending; call it once training returns. Both are no-ops without an active
    W&B run.
    """
    rows: list[list[Any]] = []
    state: dict[str, int | None] = {"step": None}

    def flush() -> None:
        if not rows:
            return
        try:
            import wandb
        except ImportError:
            rows.clear()
            return
        if wandb.run is not None:
            table = wandb.Table(
                columns=["step", "question", "answer", "completion", "reward"],
                data=list(rows),
            )
            wandb.run.log({"eval/rollouts": table, "global_step": state["step"]})
        rows.clear()

    def add(
        step: int,
        completions: Sequence[str],
        rewards: Sequence[float],
        **columns: Any,
    ) -> None:
        if state["step"] is not None and step != state["step"]:
            flush()
        state["step"] = step
        width = len(completions)
        question = answer = None
        if _is_per_completion_column(columns.get("question"), width):
            question = list(columns["question"])
        if _is_per_completion_column(columns.get("answer"), width):
            answer = list(columns["answer"])
        for index, completion in enumerate(completions):
            rows.append(
                [
                    step,
                    str(question[index]) if question else "",
                    str(answer[index]) if answer else "",
                    str(completion)[:max_chars],
                    float(rewards[index]),
                ]
            )

    return add, flush


def build_rollout_recorder(
    path: str, reward_fns: Sequence[Callable[..., list[float]]]
) -> Callable[..., int]:
    """Return ``record(prompts, completions, rewards, step, mode, **columns)``.

    Each call appends one JSON line per completion to ``path``: the step and
    mode, the rendered prompt, every per-completion dataset column
    (``question`` and ``answer``), the completion, the summed reward Tunix
    used, and each reward function's own score under its ``__name__``,
    recomputed from the strings.
    """
    lock = threading.Lock()

    def record(
        prompts: Sequence[str],
        completions: Sequence[str],
        rewards: Sequence[float],
        step: int | None,
        mode: str,
        **columns: Any,
    ) -> int:
        columns = {
            key: list(value)
            for key, value in columns.items()
            if _is_per_completion_column(value, len(completions))
        }
        per_fn = {
            fn.__name__: fn(prompts=prompts, completions=completions, **columns)
            for fn in reward_fns
        }
        rows = []
        for index, (prompt, completion) in enumerate(
            zip(prompts, completions, strict=True)
        ):
            row = {"step": step, "mode": mode, "prompt": prompt}
            row.update({key: value[index] for key, value in columns.items()})
            row["completion"] = completion
            row["reward"] = float(rewards[index])
            row["rewards"] = {
                name: float(scores[index]) for name, scores in per_fn.items()
            }
            rows.append(json.dumps(row, ensure_ascii=False))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with lock, open(path, "a", encoding="utf-8") as handle:
            handle.write("\n".join(rows) + "\n")
        return len(rows)

    return record


def run(config: dict[str, Any]) -> None:
    from tunix.rl import rl_cluster as rl_cluster_lib
    from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
    from tunix.rl.rollout import base_rollout
    from tunix.sft import checkpoint_options, metrics_logger

    mesh = create_mesh(config)

    # Reference: the frozen starting model, no LoRA -- GRPO's KL anchor.
    reference_config = copy.deepcopy(config)
    reference_config["model"].pop("lora_config", None)
    reference_model, tokenizer_path = create_model(reference_config, mesh)

    # Actor: the identical weights, with a fresh LoRA adapter to train.
    actor_model, _ = create_model(config, mesh)
    require_lora(actor_model)

    tokenizer = create_tokenizer(config, tokenizer_path)
    rollout = config["rollout"]
    eos_token_ids = [int(token) for token in rollout.get("eos_token_ids") or ()] or [
        assistant_turn_end_id(tokenizer)
    ]
    reward_fns = reward_fns_from_names(config["grpo"].get("reward_functions"))

    train_ds, eval_ds = load_grpo_prompts(config["dataset"], tokenizer)

    training = config["training"]
    max_steps = int(training["max_steps"])
    checkpointing = checkpoint_options.checkpointing_options_from_dict(
        training.get("checkpointing_options", {})
    )
    metrics = metrics_logger_options(config, metrics_logger)

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
            # Prompts per forward/backward pass. Below the batch size, Tunix
            # accumulates gradients; this is what bounds the logits memory
            # (sequences x length x vocabulary) for a large-vocabulary model.
            train_micro_batch_size=int(
                training.get("train_micro_batch_size", config["dataset"]["batch_size"])
            ),
            rollout_micro_batch_size=training.get("rollout_micro_batch_size"),
            compute_logps_micro_batch_size=training.get(
                "compute_logps_micro_batch_size"
            ),
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
            eos_tokens=eos_token_ids,
        ),
    )

    grpo = config["grpo"]
    # Passed only when set, so an unset mode keeps Tunix's own default.
    optional = {
        key: grpo[key]
        for key in ("loss_agg_mode", "kl_loss_mode")
        if grpo.get(key) is not None
    }
    grpo_config = GRPOConfig(
        num_generations=int(grpo["num_generations"]),
        num_iterations=int(grpo.get("num_iterations", 1)),
        beta=float(grpo.get("beta", 0.04)),
        epsilon=float(grpo.get("epsilon", 0.2)),
        **optional,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )
    eval_rollouts_path = training.get("eval_rollouts_path")
    wandb_enabled = bool(training.get("wandb", {}).get("enabled", False))
    table_add, table_flush = build_eval_table_logger()
    if eval_rollouts_path:
        record = build_rollout_recorder(eval_rollouts_path, reward_fns)

        class RecordingGRPOLearner(GRPOLearner):
            """GRPOLearner that keeps the text of every eval completion.

            ``RLLearner._compute_rewards`` is the one place in the pinned Tunix
            that sees the completion strings together with the mode; its
            signature is mirrored exactly.
            """

            def _compute_rewards(self, prompts, completions, mode, step=None, **kw):
                rewards = super()._compute_rewards(
                    prompts, completions, mode, step=step, **kw
                )
                if str(getattr(mode, "name", mode)).upper().endswith("EVAL"):
                    if step is None:
                        step = int(self.rl_engine.actor_trainer.train_steps)
                    record(prompts, completions, list(rewards), step, "eval", **kw)
                    if wandb_enabled:
                        table_add(step, completions, list(rewards), **kw)
                elif wandb_enabled:
                    # First training rollout after an eval: that eval is done.
                    table_flush()
                return rewards

        learner_cls: type[GRPOLearner] = RecordingGRPOLearner
    else:
        learner_cls = GRPOLearner

    grpo_trainer = learner_cls(
        rl_engine=rl_cluster,
        algo_config=grpo_config,
        reward_fns=reward_fns,
        metric_fns=[build_behaviour_metric_fn(grpo_config.num_generations)],
    )

    LOGGER.info(
        "Starting GRPO: model=%s mesh=%s max_steps=%d num_generations=%d beta=%s "
        "eos_token_ids=%s eval_rollouts_path=%s",
        config["model"]["model_id"],
        tuple(config["model"]["mesh"]["shape"]),
        max_steps,
        grpo_config.num_generations,
        grpo_config.beta,
        eos_token_ids,
        eval_rollouts_path,
    )
    # The actor update goes through PeftTrainer, which reads JAX's legacy
    # thread-local mesh, so jax.set_mesh(mesh) alone is not enough.
    with mesh:
        grpo_trainer.train(train_ds, eval_ds)
    table_flush()

    export_model(
        config=config,
        model=actor_model,
        tokenizer=tokenizer,
        local_model_path=local_base_model_path(config),
    )


def main() -> None:
    args = parse_recipe_args(recipe_parser(__doc__))
    config = load_config(args.config, args.overrides, validator=validate_grpo_config)
    run(config)


if __name__ == "__main__":
    main()
