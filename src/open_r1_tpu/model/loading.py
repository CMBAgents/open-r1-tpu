"""Loading a Tunix model from the Hub or an already-staged local directory."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    model = dict(config["model"])
    model.pop("mesh")
    return model


def create_model(config: dict[str, Any], mesh: Any) -> tuple[Any, str]:
    """Create a Tunix model from the Hub or an already-staged local directory."""
    import jax
    import jax.numpy as jnp
    from tunix.cli.utils import model as model_utils
    from tunix.models import automodel

    model_config = _model_config(config)
    if model_config.get("architecture") == "llama":
        from open_r1_tpu.model.llama import create_llama_model

        if model_config.get("lora_config") and config.get("export", {}).get(
            "enabled", False
        ):
            # Fail before training rather than after: there is no merged-LoRA
            # exporter for the local Llama path.
            raise NotImplementedError(
                "Merged-LoRA export is not implemented for the local Llama "
                "path. Set export.enabled=false; the Orbax LoRA checkpoint is "
                "the artifact."
            )
        model, path = create_llama_model(model_config, mesh)
        return _maybe_apply_lora(model, mesh, model_config), path
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
    return _maybe_apply_lora(model, mesh, model_config), str(local_path)


def _maybe_apply_lora(model: Any, mesh: Any, model_config: dict[str, Any]) -> Any:
    """Wrap ``model`` with the recipe's LoRA adapter, if it asks for one.

    Shared by the Tunix-native and local Llama paths, so both get the same
    adapter placement and seeding.
    """
    if not model_config.get("lora_config"):
        return model
    from tunix.cli.utils import model as model_utils

    return model_utils.apply_lora_to_model(
        model,
        mesh,
        model_config["lora_config"],
        rng_seed=int(model_config.get("rng_seed", 0)),
    )


def absolute_checkpoint_dir(checkpoint_dir: str) -> str:
    """Resolve a local checkpoint directory; Orbax rejects relative paths.

    Remote URIs such as ``gs://bucket/run`` are already absolute and must be
    passed through untouched, since resolving them against the working
    directory would corrupt the scheme.
    """
    if "://" in checkpoint_dir:
        return checkpoint_dir
    return str(Path(checkpoint_dir).expanduser().resolve())
