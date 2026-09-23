"""TensorBoard/W&B metrics logging shared by the SFT and GRPO training stages."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

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

_WANDB_METRIC_PREFIXES = ("train/", "eval/")
# Tunix's RL metrics are named "<group>/<mode>/<name>", e.g.
# "rewards/train/mean", "actor/train/kl", "completions/eval/mean_length".
_RL_EVENT = re.compile(r"^(?P<group>[^/]+)/(?P<mode>train|eval)/(?P<name>.+)$")
# jax.monitoring's own events (compile times, Orbax I/O) are not training
# metrics and carry no training step.
_EXCLUDED_GROUPS = {"jax"}
STEP_METRIC = "global_step"


class SteppedTrainingMetricsBackend:
    """Forward only logically stepped train/eval metrics to a backend.

    SFT metrics already start with ``train/`` or ``eval/``. GRPO metrics from
    Tunix's RL loop are ``<group>/<mode>/<name>``; they are renamed to
    ``<mode>/<group>/<name>`` so a dashboard groups them by train and eval
    (``train/rewards/mean``, ``eval/behaviour/answer_line_frac``). Anything
    else, and anything without a step, is dropped.
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def log_scalar(self, event: str, value: Any, **kwargs: Any) -> None:
        event_name = event.lstrip("/")
        if kwargs.get("step") is None:
            return
        if not event_name.startswith(_WANDB_METRIC_PREFIXES):
            match = _RL_EVENT.match(event_name)
            if match is None or match["group"] in _EXCLUDED_GROUPS:
                return
            event = f"/{match['mode']}/{match['group']}/{match['name']}"
        self._backend.log_scalar(event, value, **kwargs)

    def close(self) -> None:
        self._backend.close()


class StepAxisWandbBackend:
    """One W&B row per training step, plotted against ``global_step``.

    Wraps an initialised W&B backend (metrax's ``WandbBackend``, which owns
    ``wandb.init``). Its own ``log_scalar`` passes ``step=`` to
    ``wandb.log``, and W&B silently drops any row whose step is below the
    last one logged, which interleaved RL, actor and eval metrics can
    trigger. Here metrics are collected per step and written as one row
    carrying ``global_step``, declared as every metric's x-axis, so late
    arrivals still land at the right step.
    """

    def __init__(self, backend: Any):
        self._backend = backend
        self._step: int | None = None
        self._row: dict[str, float] = {}
        self._axis_defined = False

    def _wandb(self) -> Any:
        wandb = getattr(self._backend, "wandb", None)
        if wandb is None or not getattr(self._backend, "_is_active", False):
            return None
        return wandb

    def log_scalar(self, event: str, value: Any, **kwargs: Any) -> None:
        step = kwargs.get("step")
        if step is None:
            return
        step = int(step)
        if self._step is not None and step != self._step:
            self.flush()
        self._step = step
        array = np.asarray(value, dtype=np.float64)
        self._row[event.lstrip("/")] = float(array.mean()) if array.size else 0.0

    def flush(self) -> None:
        wandb = self._wandb()
        if wandb is not None and self._row:
            if not self._axis_defined:
                wandb.define_metric(STEP_METRIC)
                wandb.define_metric("*", step_metric=STEP_METRIC)
                self._axis_defined = True
            wandb.log({**self._row, STEP_METRIC: self._step})
        self._row = {}

    def close(self) -> None:
        self.flush()
        self._backend.close()


def wandb_backend_kwargs(config: dict[str, Any]) -> dict[str, Any]:
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


def metrics_logger_options(config: dict[str, Any], metrics_logger: Any) -> Any:
    """Create TensorBoard plus a step-safe, training-only W&B backend."""
    training = config["training"]
    log_dir = training["metrics_log_dir"]
    flush_every_n_steps = int(training.get("flush_every_n_steps", 20))

    def create_tensorboard_backend() -> Any:
        return metrics_logger.TensorboardBackend(
            log_dir=log_dir,
            flush_every_n_steps=flush_every_n_steps,
        )

    backend_factories = [create_tensorboard_backend]
    if training.get("wandb", {}).get("enabled", False):
        project_name = training.get("project_name", "open-r1-tpu")
        run_name = training.get("run_name", "reasoning-sft")
        wandb_kwargs = wandb_backend_kwargs(config)

        def create_wandb_backend() -> SteppedTrainingMetricsBackend:
            backend = metrics_logger.WandbBackend(
                project=project_name,
                name=run_name,
                **wandb_kwargs,
            )
            return SteppedTrainingMetricsBackend(StepAxisWandbBackend(backend))

        backend_factories.append(create_wandb_backend)

    return metrics_logger.MetricsLoggerOptions(
        log_dir=log_dir,
        project_name=training.get("project_name", "open-r1-tpu"),
        run_name=training.get("run_name", "reasoning-sft"),
        flush_every_n_steps=flush_every_n_steps,
        backend_kwargs={"custom_backend": backend_factories},
    )
