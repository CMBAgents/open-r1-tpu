"""Full-parameter and merged-LoRA safetensors export for Tunix models.

The pinned Tunix only ships ``save_lora_merged_model_as_safetensors``, which
adds LoRA deltas to the base checkpoint. A full fine-tune has no adapters, so
this module walks the live parameters instead and writes them back under
Hugging Face names, inverting the loader's key and transform mapping in
``tunix/models/<family>/params.py``. The export is checked against the base
checkpoint's key set, so a mapping gap fails loudly instead of silently
dropping a trained tensor.

Qwen2 and Qwen3 share every rule that touches a tensor's shape and differ only
in which parameters exist: Qwen3 has per-head query and key norms, Qwen2 has
query, key and value biases. A model with tied embeddings has no ``lm_head``
at all, and the key-set check proves the export agrees.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from open_r1_tpu.model.tokenizing import assistant_turn_end_id

LOGGER = logging.getLogger(__name__)

# Every `export` key export_model reads.
EXPORT_KEYS = frozenset({"enabled", "output_dir", "overwrite"})

# One live parameter path and value in, one Hugging Face safetensors entry out.
SafetensorsEntryFn = Callable[[str, np.ndarray], "tuple[str, np.ndarray]"]

_ATTN_QKV = re.compile(r"^layers\.(\d+)\.attn\.([qkv]_proj)\.w$")
_ATTN_OUT = re.compile(r"^layers\.(\d+)\.attn\.o_proj\.w$")
_MLP = re.compile(r"^layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.kernel$")
_LAYER_NORM = re.compile(
    r"^layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.w$"
)
# Qwen3 only: per-head query and key norms, which Qwen2 does not have.
_ATTN_NORM = re.compile(r"^layers\.(\d+)\.attn\.(q_norm|k_norm)\.w$")
# Qwen2 only: projection biases, which Qwen3 does not have. The loader stores
# them flat, exactly as the checkpoint holds them, because the attention block
# adds them after reshaping the projection back to (batch, time, heads*dim).
_ATTN_BIAS = re.compile(r"^layers\.(\d+)\.attn\.([qkv])_bias$")


def _shared_safetensors_entry(
    path: str, value: np.ndarray
) -> tuple[str, np.ndarray] | None:
    """Map one parameter under the rules both families share, else None.

    Inverts the loader's transform rules: q/k/v_proj are stored as
    (embed, heads, head_dim) and become (heads*head_dim, embed); o_proj is
    (heads, head_dim, embed) and becomes (embed, heads*head_dim); MLP kernels
    and lm_head transpose; norms and the embedding pass through unchanged.
    """
    if path == "embedder.input_embedding":
        return "model.embed_tokens.weight", value

    match = _ATTN_QKV.match(path)
    if match:
        layer, name = match.groups()
        if value.ndim != 3:
            raise ValueError(f"{path}: expected a 3D tensor, got shape {value.shape}")
        embed_dim = value.shape[0]
        flat = value.reshape(embed_dim, -1)
        return f"model.layers.{layer}.self_attn.{name}.weight", flat.transpose(1, 0)

    match = _ATTN_OUT.match(path)
    if match:
        if value.ndim != 3:
            raise ValueError(f"{path}: expected a 3D tensor, got shape {value.shape}")
        embed_dim = value.shape[-1]
        flat = value.reshape(-1, embed_dim)
        layer = match.group(1)
        return f"model.layers.{layer}.self_attn.o_proj.weight", flat.transpose(1, 0)

    match = _MLP.match(path)
    if match:
        layer, name = match.groups()
        return f"model.layers.{layer}.mlp.{name}.weight", value.transpose(1, 0)

    match = _LAYER_NORM.match(path)
    if match:
        layer, name = match.groups()
        return f"model.layers.{layer}.{name}.weight", value

    if path == "final_norm.w":
        return "model.norm.weight", value
    if path == "lm_head.w":
        return "lm_head.weight", value.transpose(1, 0)

    return None


def qwen3_safetensors_entry(path: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    """Map one live Tunix Qwen3 parameter to its safetensors entry."""
    entry = _shared_safetensors_entry(path, value)
    if entry is not None:
        return entry

    match = _ATTN_NORM.match(path)
    if match:
        layer, name = match.groups()
        return f"model.layers.{layer}.self_attn.{name}.weight", value

    raise ValueError(f"No safetensors mapping for Qwen3 parameter {path!r}")


def qwen2_safetensors_entry(path: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    """Map one live Tunix Qwen2 parameter to its safetensors entry."""
    entry = _shared_safetensors_entry(path, value)
    if entry is not None:
        return entry

    match = _ATTN_BIAS.match(path)
    if match:
        layer, name = match.groups()
        return f"model.layers.{layer}.self_attn.{name}_proj.bias", value

    raise ValueError(f"No safetensors mapping for Qwen2 parameter {path!r}")


SAFETENSORS_ENTRY_FNS: dict[str, SafetensorsEntryFn] = {
    "qwen2": qwen2_safetensors_entry,
    "qwen3": qwen3_safetensors_entry,
}


def safetensors_entry_fn(model_name: str) -> SafetensorsEntryFn:
    """Pick the parameter mapping for a Tunix model name.

    The family is read off the module Tunix itself loads rather than guessed
    from the name, which does not always carry it:
    ``deepseek-r1-distill-qwen-1.5b`` is a Qwen2 architecture.
    """
    from tunix.models import automodel

    module = automodel.get_model_module(model_name, automodel.ModelModule.MODEL)
    family = next(
        (part for part in module.__name__.split(".") if part in SAFETENSORS_ENTRY_FNS),
        None,
    )
    if family is None:
        raise NotImplementedError(
            "Full-model safetensors export is not implemented for "
            f"{model_name} ({module.__name__}). Implemented architectures: "
            f"{', '.join(sorted(SAFETENSORS_ENTRY_FNS))}. Either disable "
            "export.enabled or add a mapping to open_r1_tpu.model.export."
        )
    return SAFETENSORS_ENTRY_FNS[family]


# Merged-LoRA export for Qwen2, which the pinned Tunix's qwen2/params.py lacks.
# Tunix's generic saver does the merge given the adapter-path-to-key rule and
# the transposes, and both are Qwen3's: the two families name and lay out their
# projections identically.
QWEN_LORA_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
_QWEN_LORA_TRANSPOSE_RULES = dict.fromkeys(QWEN_LORA_MODULES, (1, 0))


def qwen_lora_state_key(lora_path: str) -> str:
    """``layers.0.attn.q_proj`` -> ``model.layers.0.self_attn.q_proj.weight``."""
    return f"model.{lora_path}.weight".replace(".attn.", ".self_attn.")


def save_qwen2_lora_merged_model_as_safetensors(
    *,
    local_model_path: str,
    output_dir: str,
    lora_model: Any,
    rank: int,
    alpha: float,
) -> None:
    """Merge a Qwen2 LoRA adapter into the base checkpoint and save it."""
    from tunix.models import safetensors_saver

    safetensors_saver.save_lora_merged_model_as_safetensors(
        local_model_path=local_model_path,
        output_dir=output_dir,
        lora_model=lora_model,
        rank=rank,
        alpha=alpha,
        state_key_transform_fn=qwen_lora_state_key,
        transpose_rules=_QWEN_LORA_TRANSPOSE_RULES,
    )


def collect_safetensors_state(
    named_params: list[tuple[str, np.ndarray]],
    entry_fn: SafetensorsEntryFn,
) -> dict[str, np.ndarray]:
    """Map every live parameter, rejecting duplicates."""
    exported: dict[str, np.ndarray] = {}
    for path, value in named_params:
        key, tensor = entry_fn(path, value)
        if key in exported:
            raise ValueError(f"Duplicate safetensors key {key!r} from {path!r}")
        exported[key] = tensor
    return exported


def export_full_model(
    *,
    model: Any,
    local_model_path: str,
    output_dir: str,
    model_name: str,
) -> None:
    """Write the model's live parameters as an unsharded HF checkpoint.

    The key set must match the base checkpoint's exactly: a missing key is a
    gap in the mapping, an extra one a diverged architecture (an untied
    lm_head the base never had). Either aborts the export.
    """
    import jax.numpy as jnp
    import safetensors.flax as safe_flax
    from flax import nnx

    # Resolved first, so an unsupported architecture fails before any work.
    entry_fn = safetensors_entry_fn(model_name)

    named_params: list[tuple[str, np.ndarray]] = []
    state = nnx.state(model, nnx.Param)
    for path, variable in state.flat_state():
        name = ".".join(str(part) for part in path)
        named_params.append((name, np.asarray(getattr(variable, "value", variable))))
    exported = collect_safetensors_state(named_params, entry_fn)

    base_state = safe_flax.load_file(
        os.path.join(local_model_path, "model.safetensors")
    )
    missing = sorted(set(base_state) - set(exported))
    extra = sorted(set(exported) - set(base_state))
    if missing or extra:
        raise ValueError(
            "Full-model export does not line up with the base checkpoint. "
            f"Missing keys: {missing}; unexpected keys: {extra}"
        )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    to_save = {
        key: jnp.asarray(tensor, dtype=base_state[key].dtype)
        for key, tensor in exported.items()
    }
    safe_flax.save_file(to_save, os.path.join(output_dir, "model.safetensors"))

    for filename in os.listdir(local_model_path):
        if not filename.endswith(".safetensors"):
            source = os.path.join(local_model_path, filename)
            if os.path.isfile(source):
                shutil.copy(source, os.path.join(output_dir, filename))


def write_turn_end_generation_config(*, output_dir: str, tokenizer: Any) -> None:
    """Put the assistant turn-end token first in the export's EOS ids.

    It must be a token-level EOS: serving commonly strips special tokens
    before matching stop strings, so a stop string cannot end the turn.
    """
    path = os.path.join(output_dir, "generation_config.json")
    generation_config: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as handle:
            generation_config = json.load(handle)
    turn_end = assistant_turn_end_id(tokenizer)
    existing = generation_config.get("eos_token_id", [])
    if isinstance(existing, int):
        existing = [existing]
    generation_config["eos_token_id"] = [turn_end] + [
        token for token in existing if token != turn_end
    ]
    with open(path, "w") as handle:
        json.dump(generation_config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def local_base_model_path(config: dict[str, Any]) -> str:
    """The local base-model directory a merged export starts from."""
    model = config["model"]
    if model.get("model_source") == "huggingface":
        path = model.get("model_download_path")
    else:
        path = model.get("model_path")
    if not path:
        raise ValueError("No local base-model path is available for merged export")
    return path


def merged_lora_saver(model_name: str) -> Callable[..., Any]:
    """The merged-LoRA safetensors saver for a Tunix model: Tunix's own where
    its params module has one, this module's for Qwen2. Raises
    NotImplementedError otherwise, so preflight can call it before training.
    """
    from tunix.models import automodel

    params_module = automodel.get_model_module(model_name, automodel.ModelModule.PARAMS)
    save_fn = getattr(params_module, "save_lora_merged_model_as_safetensors", None)
    if save_fn is None:
        try:
            is_qwen2 = safetensors_entry_fn(model_name) is qwen2_safetensors_entry
        except NotImplementedError:
            is_qwen2 = False
        if is_qwen2:
            save_fn = save_qwen2_lora_merged_model_as_safetensors
    if save_fn is None:
        raise NotImplementedError(
            "This Tunix model does not expose merged-LoRA safetensors "
            "export. Disable export.enabled or choose a supported model "
            "such as Qwen2 or Qwen3."
        )
    return save_fn


def export_model(
    *,
    config: dict[str, Any],
    model: Any,
    tokenizer: Any,
    local_model_path: str,
) -> None:
    """Export the trained model as safetensors if ``export.enabled``.

    A LoRA run merges its adapter into the base checkpoint at
    `local_model_path`; a full fine-tune writes its live parameters. The
    output directory may never be, or contain, the filesystem root, the home
    or working directory, the base model or the checkpoint directory.
    """
    export = config.get("export", {})
    if not export.get("enabled", False):
        return

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
        raise ValueError(f"Refusing unsafe merged export directory: {output_path}")
    if output_path.exists() and not export.get("overwrite", False):
        raise FileExistsError(
            f"Merged export directory already exists: {output_path}. Set "
            "export.overwrite=true to replace it."
        )
    output_dir = str(output_path)
    lora = config["model"].get("lora_config")
    if lora:
        save_fn = merged_lora_saver(str(config["model"]["model_name"]))
        LOGGER.info("Exporting merged LoRA model to %s", output_dir)
        save_fn(
            local_model_path=local_model_path,
            output_dir=output_dir,
            lora_model=model,
            rank=int(lora["rank"]),
            alpha=float(lora["alpha"]),
        )
    else:
        LOGGER.info("Exporting full fine-tuned model to %s", output_dir)
        export_full_model(
            model=model,
            local_model_path=local_model_path,
            output_dir=output_dir,
            model_name=str(config["model"]["model_name"]),
        )

    write_turn_end_generation_config(output_dir=output_dir, tokenizer=tokenizer)
