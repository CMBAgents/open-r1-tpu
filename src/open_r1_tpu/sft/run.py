"""Supervised reasoning distillation with Tunix.

Run with::

  python -m open_r1_tpu.sft.run --config \
    recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
"""

from __future__ import annotations

import logging
from typing import Any

from open_r1_tpu.core.cli import parse_recipe_args, recipe_parser
from open_r1_tpu.core.config import load_config
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
from open_r1_tpu.sft import transcripts
from open_r1_tpu.sft.config import validate_sft_config
from open_r1_tpu.sft.data import load_reasoning_datasets

LOGGER = logging.getLogger(__name__)


def _compute_max_steps(
    config: dict[str, Any], raw_train_size: int | None = None
) -> int:
    configured = config["training"].get("max_steps")
    if configured is not None:
        return int(configured)
    if raw_train_size is None:
        raise ValueError("training.max_steps is required when dataset size is unknown")

    batch_size = int(config["dataset"]["batch_size"])
    epochs = int(config["dataset"].get("num_train_epochs", 1))
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    micro_batches = (raw_train_size // batch_size) * epochs
    return max(1, micro_batches // accumulation)


def run(config: dict[str, Any]) -> None:
    import jax.numpy as jnp
    import optax
    from tunix.sft import checkpoint_options, metrics_logger, peft_trainer
    from tunix.sft import utils as sft_utils

    mesh = create_mesh(config)
    model, tokenizer_path = create_model(config, mesh)
    if config["model"].get("lora_config"):
        require_lora(model)
    tokenizer = create_tokenizer(config, tokenizer_path)

    train_ds, eval_ds = load_reasoning_datasets(config["dataset"], tokenizer)
    max_examples = config["dataset"].get("max_examples")
    # If max_examples is omitted, load_reasoning_datasets uses the full split
    # and the recipe must provide max_steps.
    max_steps = _compute_max_steps(
        config, int(max_examples) if max_examples is not None else None
    )

    training = config["training"]
    checkpointing = checkpoint_options.checkpointing_options_from_dict(
        training.get("checkpointing_options", {})
    )
    metrics = metrics_logger_options(config, metrics_logger)
    trainer = peft_trainer.PeftTrainer(
        model,
        create_optimizer(config, max_steps),
        peft_trainer.TrainingConfig(
            eval_every_n_steps=int(training.get("eval_every_n_steps", 100)),
            max_steps=max_steps,
            gradient_accumulation_steps=int(
                training.get("gradient_accumulation_steps", 1)
            ),
            checkpoint_root_directory=absolute_checkpoint_dir(
                training["checkpoint_dir"]
            ),
            checkpointing_options=checkpointing,
            metrics_logging_options=metrics,
            profiler_options=None,
            data_sharding_axis=tuple(training.get("data_sharding_axis", ["fsdp"])),
            max_inflight_computations=int(training.get("max_inflight_computations", 1)),
        ),
    )

    def gen_model_input(training_input):
        if isinstance(training_input, dict):
            # Packed batch: the packer already carries per-segment positions
            # and segment ids (1..K per example, 0 on padding). segment_ids
            # gate the splash kernel; the block-diagonal causal mask serves
            # the non-flash attention path, which ignores segment_ids.
            segment_ids = jnp.asarray(training_input["segment_ids"])
            seq_len = segment_ids.shape[-1]
            causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
            same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
            return {
                "input_tokens": training_input["input_tokens"],
                "input_mask": training_input["input_mask"],
                "positions": training_input["positions"],
                "attention_mask": same_segment & causal[None, ...],
                "segment_ids": segment_ids,
            }
        # Derive sequence lengths from the target mask instead of comparing
        # token IDs. Some chat tokenizers use EOS as PAD, so ID comparison
        # would incorrectly hide real in-sequence end-of-turn tokens.
        token_positions = jnp.arange(training_input.input_mask.shape[-1])
        last_target = jnp.max(
            jnp.where(training_input.input_mask, token_positions, -1), axis=-1
        )
        pad_mask = token_positions[None, :] <= last_target[:, None]
        return {
            "input_tokens": training_input.input_tokens,
            "input_mask": training_input.input_mask,
            "positions": sft_utils.build_positions_from_mask(pad_mask),
            "attention_mask": sft_utils.make_causal_attn_mask(pad_mask),
        }

    def sparse_causal_lm_loss(
        model,
        input_tokens,
        input_mask,
        positions,
        attention_mask,
        segment_ids=None,
    ):
        """Masked causal loss on integer labels, not a vocabulary-sized one-hot.

        Returned as a sum and a token count rather than a mean, so gradient
        accumulation weights every supervised token equally.
        """
        logits, _ = model(
            input_tokens, positions, None, attention_mask, segment_ids=segment_ids
        )
        token_loss = optax.softmax_cross_entropy_with_integer_labels(
            logits[:, :-1, :], input_tokens[:, 1:]
        )
        target_mask = input_mask[:, 1:].astype(token_loss.dtype)
        return sft_utils.LossOutput(
            primary_loss=sft_utils.WeightedMetric(
                unreduced_sum=jnp.sum(token_loss * target_mask),
                denominator=jnp.sum(target_mask),
                eps=1e-8,
            ),
            aux_metrics={},
        )

    trainer = trainer.with_gen_model_input_fn(gen_model_input).with_loss_fn(
        sparse_causal_lm_loss
    )

    transcript_settings = transcripts.resolve_settings(config)
    if transcript_settings["enabled"]:
        LOGGER.info(
            "Transcripts enabled: %d prompt(s) every %d steps to %s",
            len(transcript_settings["prompts"]),
            transcript_settings["every_n_steps"],
            transcript_settings["output_path"],
        )
        # Unlike with_loss_fn, with_training_hooks returns None rather than the
        # trainer, so it must not be chained.
        trainer.with_training_hooks(
            transcripts.create_training_hooks(model, tokenizer, transcript_settings)
        )

    LOGGER.info(
        "Starting reasoning SFT: model=%s mesh=%s max_steps=%d",
        config["model"]["model_id"],
        tuple(config["model"]["mesh"]["shape"]),
        max_steps,
    )
    # PeftTrainer reads JAX's legacy thread-local mesh, so jax.set_mesh(mesh)
    # alone is not enough.
    with mesh:
        trainer.train(train_ds, eval_ds)

    export_model(
        config=config,
        model=model,
        tokenizer=tokenizer,
        local_model_path=local_base_model_path(config),
    )


def main() -> None:
    args = parse_recipe_args(recipe_parser(__doc__))
    config = load_config(args.config, args.overrides, validator=validate_sft_config)
    run(config)


if __name__ == "__main__":
    main()
