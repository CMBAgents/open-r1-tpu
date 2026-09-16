"""Optimizer construction shared by the SFT and GRPO training stages."""

from __future__ import annotations

from typing import Any


def create_optimizer(config: dict[str, Any], max_steps: int):
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
