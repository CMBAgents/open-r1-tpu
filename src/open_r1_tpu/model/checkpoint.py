"""Load a staged model or a training checkpoint onto the TPUs outside training.

``scripts/chat_tpu.py`` and ``scripts/complete_tpu.py`` load a local model onto
every visible chip for sampling, optionally restoring a recipe's checkpoint on
top; ``scripts/export_checkpoint.py`` restores a checkpoint for export. JAX and
Tunix are imported inside the functions that use them, so importing this module
does not initialise a TPU.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

FLASH_ATTENTION_BLOCK_SIZE = 1024


class SamplerRuntime(NamedTuple):
    """A loaded model's mesh, tokenizer and Tunix sampler.

    ``fsdp_size`` is the mesh's FSDP width: every sampler batch must be padded
    to a multiple of it (see `pad_input_strings_for_fsdp`).
    """

    mesh: Any
    tokenizer: Any
    sampler: Any
    fsdp_size: int


@contextmanager
def tunix_mesh_context(mesh: Any) -> Iterator[None]:
    """Activate `mesh` as JAX's legacy physical mesh for the pinned Tunix.

    Its sampler resolves ``P("tp")`` against that thread-local mesh, which
    ``jax.set_mesh`` leaves empty, so the deprecated ``with mesh:`` is still
    needed; its warning is silenced.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`with mesh:` context manager has been deprecated.*",
            category=DeprecationWarning,
        )
        with mesh:
            yield


def resolve_model_dir(model_path: str) -> str:
    """Return the absolute model directory, failing before an expensive load
    when it holds no ``model.safetensors``."""
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {path}")
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No model.safetensors found in local model directory: {path}"
        )
    return str(path)


def model_settings_for_path(
    model_path: str, model_name_override: str | None = None
) -> tuple[str, int, float | None]:
    """Read the Tunix model name, KV-head count and RoPE theta from config.json.

    Tunix names a model by the lowercase last component of the source id that
    Hugging Face keeps in ``_name_or_path``; `model_name_override` covers
    exported configs without one. ``rope_theta`` is returned because Tunix
    otherwise takes it from its registered config for that name, which would
    silently serve a re-based model at the stock theta.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No config.json found in local model directory: {config_path.parent}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in local model config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Local model config must be a JSON object: {config_path}")

    source_name = model_name_override or config.get("_name_or_path")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError(
            f"Local model config {config_path} has no _name_or_path; pass "
            "--model-name with the canonical Tunix model name."
        )
    num_kv_heads = config.get("num_key_value_heads")
    if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        raise ValueError(
            f"Local model config {config_path} has invalid num_key_value_heads: "
            f"{num_kv_heads!r}"
        )
    rope_theta = config.get("rope_theta")
    if rope_theta is not None and (
        not isinstance(rope_theta, (int, float)) or rope_theta <= 0
    ):
        raise ValueError(
            f"Local model config {config_path} has invalid rope_theta: {rope_theta!r}"
        )
    return (
        source_name.rsplit("/", maxsplit=1)[-1].lower(),
        num_kv_heads,
        float(rope_theta) if rope_theta is not None else None,
    )


def mesh_shape_for_devices(
    devices: Sequence[Any], num_kv_heads: int
) -> tuple[int, int]:
    """Use every visible TPU through FSDP and tensor parallelism.

    Tensor parallelism cannot exceed the model's KV-head count. Remaining
    chips form the FSDP axis; callers pad the sampler batch to that width.
    """
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if not devices:
        raise RuntimeError(
            "This client needs at least one visible TPU device; JAX sees no "
            "devices. Run it on a TPU VM with the TPU runtime configured."
        )
    if any(device.platform != "tpu" for device in devices):
        visible = ", ".join(f"{device.platform}:{device.id}" for device in devices)
        raise RuntimeError(
            "This client requires every visible JAX device to be a TPU; JAX sees "
            f"[{visible}]. Run it on a TPU VM with the TPU runtime configured."
        )
    tp_size = min(len(devices), num_kv_heads)
    if num_kv_heads % tp_size != 0 or len(devices) % tp_size != 0:
        raise RuntimeError(
            "This client needs a visible TPU-chip count that can form a tensor-"
            f"parallel width dividing its {num_kv_heads} KV heads; JAX sees "
            f"{len(devices)} TPU devices."
        )
    return len(devices) // tp_size, tp_size


def inference_model_config(
    model_path: str,
    seed: int,
    *,
    model_name: str,
    mesh_shape: tuple[int, int],
    lora_config: dict[str, Any] | None = None,
    rope_theta: float | None = None,
) -> dict[str, Any]:
    """Return the ``model`` section `create_model` needs for sampling."""
    config: dict[str, Any] = {
        "model_name": model_name,
        "model_source": "local",
        "model_path": model_path,
        "rng_seed": seed,
        # float32, not training's bfloat16: under the pinned Tunix the jitted
        # bfloat16 Qwen2 forward returns all-NaN logits once the KV cache
        # reaches 1536 slots, and the sampler then emits token 0 ("!")
        # forever. Reasoning replies need far longer caches.
        "dtype": "float32",
        "load_dtype": "float32",
        "remat_config": "DECODER",
        # The sampler left-pads prompts but does not pass the segment IDs that
        # splash attention needs to mask padding, so splash would let pad/EOS
        # embeddings reach real tokens. Ordinary attention uses the pad mask.
        "use_flash_attention": False,
        "flash_attention_block_size": FLASH_ATTENTION_BLOCK_SIZE,
        "mesh": {"shape": list(mesh_shape), "axis_names": ["fsdp", "tp"]},
    }
    if rope_theta is not None:
        config["rope_theta"] = rope_theta
    if lora_config:
        # Adapters must be restored into the geometry they were trained under,
        # so this comes from the recipe rather than defaults of its own.
        config["lora_config"] = lora_config
    return config


def tokenizer_config(model_path: str) -> dict[str, Any]:
    """Use the tokenizer staged beside the model, without a Hub lookup."""
    return {
        "tokenizer_path": model_path,
        "tokenizer_type": "huggingface",
        "add_bos": False,
        "add_eos": False,
        "chat_template": None,
    }


def pad_input_strings_for_fsdp(input_strings: list[str], fsdp_size: int) -> list[str]:
    """Repeat the final prompt so the sampler batch divides the FSDP width."""
    if fsdp_size <= 0:
        raise ValueError("FSDP mesh size must be positive")
    if not input_strings:
        raise ValueError("at least one input string is required")
    remainder = len(input_strings) % fsdp_size
    if remainder == 0:
        return input_strings
    return [
        *input_strings,
        *([input_strings[-1]] * (fsdp_size - remainder)),
    ]


def recipe_restore_settings(recipe_path: str) -> tuple[dict[str, Any] | None, str]:
    """Read an SFT recipe's LoRA geometry (None for a full fine-tune) and
    checkpoint root."""
    from open_r1_tpu.core.config import load_config

    config = load_config(recipe_path)
    return config["model"].get("lora_config"), config["training"]["checkpoint_dir"]


def available_steps(checkpoint_root: str) -> list[int]:
    """List the steps written under a checkpoint root, newest last."""
    root_path = Path(checkpoint_root)
    if not root_path.is_dir():
        return []
    return sorted(
        int(entry.name)
        for entry in root_path.iterdir()
        if entry.is_dir() and entry.name.isdigit()
    )


def resolve_step(checkpoint_root: str, step: int | None) -> int | None:
    """Reject a step that was never written, naming the ones that were.

    A run stopped between saves has no checkpoint at the step its log last
    reported, which is the step someone naturally asks for.
    """
    if step is None:
        return None
    steps = available_steps(checkpoint_root)
    if steps and step not in steps:
        written = ", ".join(str(value) for value in steps)
        raise FileNotFoundError(
            f"No checkpoint at step {step} under {checkpoint_root}. Steps "
            f"written and still kept: {written}. Training saves every "
            "training.checkpointing_options.save_interval_steps steps and "
            "keeps max_to_keep of them, so a run stopped between saves has "
            "no checkpoint at the step it stopped on."
        )
    return step


def restore_checkpoint(
    model: Any, checkpoint_dir: str, step: int | None = None, *, lora_only: bool
) -> int:
    """Restore a training checkpoint into `model` and return the step restored.

    `step` defaults to the latest written. `lora_only` must match what
    training saved: a LoRA run saves adapters alone, a full fine-tune every
    parameter.
    """
    from tunix.sft import checkpoint_manager as checkpoint_manager_lib

    from open_r1_tpu.model.loading import absolute_checkpoint_dir

    root = absolute_checkpoint_dir(checkpoint_dir)
    step = resolve_step(root, step)
    # Tunix's default options, not the recipe's: this only reads, and the
    # recipe's max_to_keep is a deletion policy that must not run from here.
    manager = checkpoint_manager_lib.CheckpointManager(root_directory=root)
    if manager.latest_step() is None:
        raise FileNotFoundError(
            f"No checkpoint has been written under {root}. Training writes "
            "one every training.checkpointing_options.save_interval_steps "
            "optimizer steps."
        )
    restored_step, _metadata = manager.maybe_restore(
        model, optimizer=None, step=step, restore_only_lora_params=lora_only
    )
    manager.close()
    return int(restored_step)


def load_sampler(
    model_path: str,
    *,
    seed: int,
    cache_size: int,
    model_name: str | None = None,
    recipe: str | None = None,
    checkpoint_dir: str | None = None,
    step: int | None = None,
) -> SamplerRuntime:
    """Load a local model onto every visible TPU and build a Tunix sampler.

    With `recipe`, its checkpoint at `step` (default: the latest) is restored
    on top of the base weights first; `checkpoint_dir` overrides the recipe's
    ``training.checkpoint_dir``. `cache_size` is the KV-cache length: prompt
    plus generated tokens.
    """
    import jax
    from tunix.cli.utils import model as model_utils
    from tunix.generate import sampler as sampler_lib
    from tunix.utils import mesh as mesh_utils

    from open_r1_tpu.model.loading import create_model

    tunix_name, num_kv_heads, rope_theta = model_settings_for_path(
        model_path, model_name
    )
    mesh_shape = mesh_shape_for_devices(tuple(jax.devices()), num_kv_heads)
    mesh = mesh_utils.create_mesh(mesh_shape, ("fsdp", "tp"))

    lora_config = None
    restore_dir = None
    if recipe:
        lora_config, recipe_checkpoint_dir = recipe_restore_settings(recipe)
        restore_dir = checkpoint_dir or recipe_checkpoint_dir

    config = {
        "model": inference_model_config(
            model_path,
            seed,
            model_name=tunix_name,
            mesh_shape=mesh_shape,
            lora_config=lora_config,
            rope_theta=rope_theta,
        ),
        "tokenizer": tokenizer_config(model_path),
    }
    model, tokenizer_path = create_model(config, mesh)
    if restore_dir is not None:
        restored = restore_checkpoint(
            model, restore_dir, step, lora_only=bool(lora_config)
        )
        # Printed because it is rarely the step the run stopped on.
        what = "LoRA adapters" if lora_config else "full model parameters"
        print(f"Restored {what} from step {restored}.")
    tokenizer = model_utils.create_tokenizer(config["tokenizer"], tokenizer_path)
    model_runtime_config = getattr(model, "config", None)
    if model_runtime_config is None:
        raise RuntimeError("model exposes no config; cannot size the KV cache")
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=cache_size,
            num_layers=int(model_runtime_config.num_layers),
            num_kv_heads=int(model_runtime_config.num_kv_heads),
            head_dim=int(model_runtime_config.head_dim),
        ),
    )
    return SamplerRuntime(mesh, tokenizer, sampler, mesh_shape[0])
