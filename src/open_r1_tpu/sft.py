"""TPU-native supervised reasoning distillation with Tunix.

Run with::

  python -m open_r1_tpu.sft --config \
    recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import math
from pathlib import Path
from typing import Any

from open_r1_tpu.config import load_config
from open_r1_tpu.data import load_reasoning_datasets


LOGGER = logging.getLogger(__name__)

_WANDB_INIT_KEYS = {
    "entity",
    "group",
    "job_type",
    "mode",
    "notes",
    "resume",
    "save_code",
    "tags",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML recipe path")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Tunix-style overrides such as dataset.max_examples=128",
    )
    return parser.parse_args()


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    model.pop("mesh")
    return model


def _create_model(config: dict[str, Any], mesh: Any) -> tuple[Any, str]:
    """Create a Tunix model from the Hub or an already-staged local directory."""
    import jax
    import jax.numpy as jnp
    from tunix.cli.utils import model as model_utils
    from tunix.models import automodel

    model_config = _model_config(config)
    if model_config["model_source"] != "local":
        return model_utils.create_model(model_config, config["tokenizer"], mesh)

    local_path = Path(model_config.get("model_path", "")).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"Local model directory does not exist: {local_path}")
    if not (local_path / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"Local model directory has no model.safetensors: {local_path}"
        )

    model_name = model_config["model_name"]
    model_params = automodel.call_model_config(model_name)
    valid_fields = {field.name for field in dataclasses.fields(model_params)}
    overrides = {
        key: value
        for key, value in model_config.items()
        if key in valid_fields and value is not None
    }
    if isinstance(overrides.get("remat_config"), str):
        model_module = automodel.get_model_module(
            model_name, automodel.ModelModule.MODEL
        )
        try:
            overrides["remat_config"] = getattr(
                model_module.RematConfig, overrides["remat_config"]
            )
        except AttributeError as exc:
            raise ValueError(
                f"Invalid remat_config: {overrides['remat_config']}"
            ) from exc
    if isinstance(overrides.get("dtype"), str):
        try:
            overrides["dtype"] = getattr(jnp, overrides["dtype"])
        except AttributeError as exc:
            raise ValueError(f"Invalid dtype: {overrides['dtype']}") from exc
    if overrides:
        model_params = dataclasses.replace(model_params, **overrides)

    load_dtype = model_config.get("load_dtype")
    if isinstance(load_dtype, str):
        try:
            load_dtype = getattr(jnp, load_dtype)
        except AttributeError as exc:
            raise ValueError(f"Invalid load_dtype: {load_dtype}") from exc
    with jax.set_mesh(mesh):
        model = automodel.create_model_from_safe_tensors(
            model_name,
            str(local_path),
            model_params,
            mesh,
            dtype=load_dtype,
        )
    if model_config.get("lora_config"):
        model = model_utils.apply_lora_to_model(
            model,
            mesh,
            model_config["lora_config"],
            rng_seed=int(model_config.get("rng_seed", 0)),
        )
    return model, str(local_path)


def _compute_max_steps(config: dict[str, Any], raw_train_size: int | None = None) -> int:
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


def _wandb_backend_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Build W&B initialization arguments without putting credentials in config."""
    training = config["training"]
    wandb_config = training.get("wandb", {})
    if not wandb_config.get("enabled", False):
        return {"mode": "disabled"}

    kwargs = {
        key: value
        for key, value in wandb_config.items()
        if key in _WANDB_INIT_KEYS and value is not None
    }
    kwargs.setdefault("mode", "online")
    kwargs.setdefault("save_code", True)
    kwargs["dir"] = training["metrics_log_dir"]
    # The recipe contains no secrets, so recording it makes runs reproducible.
    kwargs["config"] = config
    return kwargs


def _create_optimizer(config: dict[str, Any], max_steps: int):
    import optax

    optimizer = config["optimizer"]
    learning_rate = float(optimizer["learning_rate"])
    warmup_steps = int(max_steps * float(optimizer.get("warmup_ratio", 0.0)))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(max_steps, warmup_steps + 1),
        end_value=learning_rate * float(optimizer.get("min_lr_ratio", 0.1)),
    )
    adamw = optax.adamw(
        learning_rate=schedule,
        b1=float(optimizer.get("b1", 0.9)),
        b2=float(optimizer.get("b2", 0.99)),
        eps=float(optimizer.get("eps", 1e-8)),
        weight_decay=float(optimizer.get("weight_decay", 0.0)),
    )
    max_grad_norm = optimizer.get("max_grad_norm")
    if max_grad_norm is None:
        return adamw
    return optax.chain(optax.clip_by_global_norm(float(max_grad_norm)), adamw)


def _export_merged_lora(
    *,
    config: dict[str, Any],
    model: Any,
    local_model_path: str,
) -> None:
    export = config.get("export", {})
    if not export.get("enabled", False):
        return

    from tunix.models import automodel

    params_module = automodel.get_model_module(
        config["model"]["model_name"], automodel.ModelModule.PARAMS
    )
    save_fn = getattr(params_module, "save_lora_merged_model_as_safetensors", None)
    if save_fn is None:
        raise NotImplementedError(
            "This Tunix model does not expose merged-LoRA safetensors export. "
            "Disable export.enabled or choose a supported model such as Qwen3."
        )

    output_path = Path(export["output_dir"]).expanduser().resolve()
    protected_paths = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        Path(local_model_path).expanduser().resolve(),
        Path(config["training"]["checkpoint_dir"]).expanduser().resolve(),
    }
    if output_path in protected_paths or any(
        protected.is_relative_to(output_path) for protected in protected_paths
    ):
        raise ValueError(
            f"Refusing unsafe merged export directory: {output_path}"
        )
    if output_path.exists() and not export.get("overwrite", False):
        raise FileExistsError(
            f"Merged export directory already exists: {output_path}. Set "
            "export.overwrite=true to replace it."
        )
    output_dir = str(output_path)
    lora = config["model"].get("lora_config", {})
    LOGGER.info("Exporting merged SFT model to %s", output_dir)
    save_fn(
        local_model_path=local_model_path,
        output_dir=output_dir,
        lora_model=model,
        rank=int(lora["rank"]),
        alpha=float(lora["alpha"]),
    )


def run(config: dict[str, Any]) -> None:
    import jax
    import jax.numpy as jnp
    import optax
    from tunix.cli.utils import model as model_utils
    from tunix.sft import checkpoint_options
    from tunix.sft import metrics_logger
    from tunix.sft import peft_trainer
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

    model_config = _model_config(config)
    model, tokenizer_path = _create_model(config, mesh)
    if model_config.get("lora_config") and not sft_utils.is_lora_enabled(model):
        raise RuntimeError(
            "LoRA was requested but Tunix found no matching modules. Check "
            "model.lora_config.module_path before training."
        )
    tokenizer = model_utils.create_tokenizer(
        config["tokenizer"], tokenizer_path
    )
    if config["tokenizer"].get("chat_template"):
        tokenizer.tokenizer.chat_template = config["tokenizer"]["chat_template"]

    train_ds, eval_ds = load_reasoning_datasets(config["dataset"], tokenizer)
    max_examples = config["dataset"].get("max_examples")
    # Mixture-of-Thoughts has a known finite split. If max_examples is omitted,
    # load_reasoning_datasets uses the full split and the configured recipe must
    # provide max_steps (the default recipe does).
    max_steps = _compute_max_steps(
        config, int(max_examples) if max_examples is not None else None
    )

    training = config["training"]
    checkpointing = checkpoint_options.checkpointing_options_from_dict(
        training.get("checkpointing_options", {})
    )
    metrics = metrics_logger.MetricsLoggerOptions(
        log_dir=training["metrics_log_dir"],
        project_name=training.get("project_name", "open-r1-tpu"),
        run_name=training.get("run_name", "reasoning-sft"),
        flush_every_n_steps=int(training.get("flush_every_n_steps", 20)),
        backend_kwargs={"wandb": _wandb_backend_kwargs(config)},
    )
    trainer = peft_trainer.PeftTrainer(
        model,
        _create_optimizer(config, max_steps),
        peft_trainer.TrainingConfig(
            eval_every_n_steps=int(training.get("eval_every_n_steps", 100)),
            max_steps=max_steps,
            gradient_accumulation_steps=int(
                training.get("gradient_accumulation_steps", 1)
            ),
            checkpoint_root_directory=training["checkpoint_dir"],
            checkpointing_options=checkpointing,
            metrics_logging_options=metrics,
            profiler_options=None,
            data_sharding_axis=tuple(training.get("data_sharding_axis", ["fsdp"])),
            max_inflight_computations=int(
                training.get("max_inflight_computations", 1)
            ),
        ),
    )

    def gen_model_input(training_input):
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
    ):
        """Assistant-only causal loss without a vocabulary-sized one-hot target."""
        logits, _ = model(input_tokens, positions, None, attention_mask)
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
    LOGGER.info(
        "Starting reasoning SFT: model=%s mesh=%s max_steps=%d",
        config["model"]["model_id"],
        mesh_shape,
        max_steps,
    )
    # Tunix 0.1.8 still reads JAX's legacy thread-local physical mesh inside
    # PeftTrainer, so jax.set_mesh(mesh) is not sufficient here yet.
    with mesh:
        trainer.train(train_ds, eval_ds)

    if model_config["model_source"] == "huggingface":
        local_model_path = model_config.get("model_download_path")
    else:
        local_model_path = model_config.get("model_path")
    if not local_model_path:
        raise ValueError("No local base-model path is available for merged export")
    _export_merged_lora(
        config=config,
        model=model,
        local_model_path=local_model_path,
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config, args.overrides)
    run(config)


if __name__ == "__main__":
    main()
