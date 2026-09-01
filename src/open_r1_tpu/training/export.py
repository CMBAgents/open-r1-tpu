"""Full-parameter safetensors export for Tunix Qwen2 and Qwen3 models.

Tunix (at the pinned commit) only ships ``save_lora_merged_model_as_safetensors``,
which starts from the base checkpoint and adds LoRA deltas. A full fine-tune has
no adapters, so this module walks the live model parameters instead and writes
them back under Hugging Face names, inverting the loader's key and transform
mapping in ``tunix/models/<family>/params.py``. The export is validated against
the base checkpoint's key set so a mapping gap fails loudly instead of silently
dropping a trained tensor.

The two families share every rule that touches a tensor's shape, and differ in
which parameters exist at all: Qwen3 carries per-head query and key norms, and
Qwen2 carries biases on its query, key and value projections. The shared rules
live in ``_shared_safetensors_entry`` so the two mappings cannot drift apart in
the parts that are meant to agree.

Tied embeddings put no ``lm_head`` in the live model and no ``lm_head.weight``
in the base checkpoint -- Qwen2.5-Math-1.5B and the Qwen3 Base models are tied,
DeepSeek-R1-Distill-Qwen-1.5B is not -- so that rule is simply never exercised
on a tied model, and the key-set check below is what proves it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Callable
from typing import Any

import numpy as np

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

    The family is read off the module Tunix itself would load rather than
    guessed from the name, because the name does not always carry it:
    ``deepseek-r1-distill-qwen-1.5b`` is a Qwen2 architecture, and Tunix
    registers new names against existing families all the time. Asking the
    registry means this cannot fall out of step with what actually loaded.
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
            "export.enabled or add a mapping to open_r1_tpu.training.export."
        )
    return SAFETENSORS_ENTRY_FNS[family]


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
    *, model: Any, local_model_path: str, output_dir: str, model_name: str
) -> None:
    """Write the model's live parameters as an unsharded HF checkpoint.

    The key set must match the base checkpoint's exactly: a missing key means
    the mapping above has a gap, an extra key means the architecture diverged
    (for example an untied lm_head the base never had). Either aborts the
    export rather than producing a checkpoint that silently loses training.
    """
    import jax.numpy as jnp
    import safetensors.flax as safe_flax
    from flax import nnx

    # Resolved before any work, so an unsupported architecture fails here
    # rather than after the parameters have been walked.
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
    """Name the chat template's turn-end token as an EOS of the export.

    The config files copied from the base model carry the base EOS only
    (Qwen3-Base: ``<|endoftext|>``), while the chat template closes every turn
    with a different token (``<|im_end|>``) that the fine-tune learns to emit.
    A server reading the copied ``generation_config.json`` therefore never
    stops at the end of a turn, and a stop *string* cannot compensate: vLLM
    matches stop strings against decoded text with special tokens stripped, so
    the turn end must be a token-level EOS. Chat-tuned releases (Qwen3
    instruct) ship exactly this list, turn-end token first.
    """
    from open_r1_tpu.training.data import assistant_turn_end_id

    turn_end = assistant_turn_end_id(tokenizer)
    path = os.path.join(output_dir, "generation_config.json")
    generation_config: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path) as handle:
            generation_config = json.load(handle)
    existing = generation_config.get("eos_token_id", [])
    if isinstance(existing, int):
        existing = [existing]
    generation_config["eos_token_id"] = [turn_end] + [
        token for token in existing if token != turn_end
    ]
    with open(path, "w") as handle:
        json.dump(generation_config, handle, indent=2, sort_keys=True)
        handle.write("\n")
