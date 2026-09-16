"""TensorBoard/W&B metrics logging shared by the SFT and GRPO training stages."""

from __future__ import annotations

from typing import Any

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


class SteppedTrainingMetricsBackend:
    """Forward only logically stepped train/eval metrics to a backend."""

    def __init__(self, backend: Any):
        self._backend = backend

    def log_scalar(self, event: str, value: Any, **kwargs: Any) -> None:
        event_name = event.lstrip("/")
        if kwargs.get("step") is None:
            return
        if not event_name.startswith(_WANDB_METRIC_PREFIXES):
            return
        self._backend.log_scalar(event, value, **kwargs)

    def close(self) -> None:
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
            return SteppedTrainingMetricsBackend(backend)

        backend_factories.append(create_wandb_backend)

    return metrics_logger.MetricsLoggerOptions(
        log_dir=log_dir,
        project_name=training.get("project_name", "open-r1-tpu"),
        run_name=training.get("run_name", "reasoning-sft"),
        flush_every_n_steps=flush_every_n_steps,
        backend_kwargs={"custom_backend": backend_factories},
    )
