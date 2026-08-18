"""Full-parameter safetensors export for Tunix Qwen3 models.

Tunix (at the pinned commit) only ships ``save_lora_merged_model_as_safetensors``,
which starts from the base checkpoint and adds LoRA deltas. A full fine-tune has
no adapters, so this module walks the live model parameters instead and writes
them back under Hugging Face names, inverting the loader's key and transform
mapping in ``tunix/models/qwen3/params.py``. The export is validated against the
base checkpoint's key set so a mapping gap fails loudly instead of silently
dropping a trained tensor.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Any

import numpy as np

_ATTN_QKV = re.compile(r"^layers\.(\d+)\.attn\.([qkv]_proj)\.w$")
_ATTN_OUT = re.compile(r"^layers\.(\d+)\.attn\.o_proj\.w$")
_MLP = re.compile(r"^layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.kernel$")
_NORM = re.compile(
    r"^layers\.(\d+)\."
    r"(attn\.q_norm|attn\.k_norm|input_layernorm|post_attention_layernorm)"
    r"\.w$"
)


def qwen3_safetensors_entry(path: str, value: np.ndarray) -> tuple[str, np.ndarray]:
    """Map one live Tunix parameter to its Hugging Face safetensors entry.

    Inverts the loader's transform rules: q/k/v_proj are stored as
    (embed, heads, head_dim) and become (heads*head_dim, embed); o_proj is
    (heads, head_dim, embed) and becomes (embed, heads*head_dim); MLP kernels
    transpose; norms and the embedding pass through unchanged.
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

    match = _NORM.match(path)
    if match:
        layer, name = match.groups()
        name = name.replace("attn.", "self_attn.")
        return f"model.layers.{layer}.{name}.weight", value

    if path == "final_norm.w":
        return "model.norm.weight", value
    if path == "lm_head.w":
        return "lm_head.weight", value.transpose(1, 0)

    raise ValueError(f"No safetensors mapping for Qwen3 parameter {path!r}")


def collect_safetensors_state(
    named_params: list[tuple[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Map every live parameter, rejecting duplicates."""
    exported: dict[str, np.ndarray] = {}
    for path, value in named_params:
        key, tensor = qwen3_safetensors_entry(path, value)
        if key in exported:
            raise ValueError(f"Duplicate safetensors key {key!r} from {path!r}")
        exported[key] = tensor
    return exported


def export_full_model(
    *, model: Any, local_model_path: str, output_dir: str
) -> None:
    """Write the model's live parameters as an unsharded HF checkpoint.

    The key set must match the base checkpoint's exactly: a missing key means
    the mapping above has a gap, an extra key means the architecture diverged
    (for example an untied lm_head the base never had). Either aborts the
    export rather than producing a checkpoint that silently loses training.
    """
    from flax import nnx
    import jax.numpy as jnp
    import safetensors.flax as safe_flax

    named_params: list[tuple[str, np.ndarray]] = []
    state = nnx.state(model, nnx.Param)
    for path, variable in state.flat_state():
        name = ".".join(str(part) for part in path)
        named_params.append((name, np.asarray(getattr(variable, "value", variable))))
    exported = collect_safetensors_state(named_params)

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
