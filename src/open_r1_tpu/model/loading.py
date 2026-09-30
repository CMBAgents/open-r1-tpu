"""Device mesh, model and tokenizer creation for the training stages."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def validate_mesh_config(model: Mapping[str, Any]) -> None:
    """Check `model.mesh`: positive sizes, and one axis name per dimension."""
    mesh = model.get("mesh") or {}
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


def create_mesh(config: dict[str, Any]) -> Any:
    """Build the recipe's device mesh, which must cover every visible device."""
    import jax
    from tunix.utils import mesh as mesh_utils

    mesh_shape = tuple(config["model"]["mesh"]["shape"])
    axis_names = tuple(config["model"]["mesh"]["axis_names"])
    if math.prod(mesh_shape) != jax.device_count():
        raise ValueError(
            f"Configured mesh {mesh_shape} needs {math.prod(mesh_shape)} devices, "
            f"but JAX sees {jax.device_count()}. Override model.mesh.shape."
        )
    return mesh_utils.create_mesh(mesh_shape, axis_names)


def create_tokenizer(config: dict[str, Any], tokenizer_path: str) -> Any:
    """Create the recipe's tokenizer, applying `tokenizer.chat_template` if set."""
    from tunix.cli.utils import model as model_utils

    tokenizer = model_utils.create_tokenizer(config["tokenizer"], tokenizer_path)
    if config["tokenizer"].get("chat_template"):
        tokenizer.tokenizer.chat_template = config["tokenizer"]["chat_template"]
    return tokenizer


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    model.pop("mesh")
    return model


def create_model(config: dict[str, Any], mesh: Any) -> tuple[Any, str]:
    """Create a Tunix model from the Hub or an already-staged local directory.

    Returns the model and the directory its tokenizer should load from.
    """
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


def require_lora(model: Any) -> None:
    """Fail when LoRA was requested but matched no module.

    Tunix would otherwise train the full model without saying so.
    """
    from tunix.sft import utils as sft_utils

    if not sft_utils.is_lora_enabled(model):
        raise RuntimeError(
            "LoRA was requested but Tunix found no matching modules. Check "
            "model.lora_config.module_path before training."
        )


def absolute_checkpoint_dir(checkpoint_dir: str) -> str:
    """Resolve a local checkpoint directory; Orbax rejects relative paths.

    Remote URIs such as ``gs://bucket/run`` pass through untouched, since
    resolving them against the working directory would corrupt the scheme.
    """
    if "://" in checkpoint_dir:
        return checkpoint_dir
    return str(Path(checkpoint_dir).expanduser().resolve())
