"""Load a local HF Llama checkpoint with Tunix's packed Qwen2 kernels.

Dense Llama with ordinary RoPE is Qwen2 without projection biases. The pinned
Tunix Llama implementation lacks flash attention and ignores its RoPE setting,
so reuse Qwen2's tested attention, replacing its biases with parameter-free
zeros. No checkpoint tensors are added and no dependency is monkey-patched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def llama_dimensions(hf: dict[str, Any]) -> dict[str, Any]:
    """Reject unsupported variants instead of silently changing the model."""
    if hf.get("model_type") != "llama":
        raise ValueError("model.architecture=llama requires a Llama config")
    if hf.get("rope_scaling") or hf.get("attention_bias") or hf.get("mlp_bias"):
        raise ValueError("Llama loader requires ordinary RoPE and no biases")
    if hf.get("hidden_act", "silu") != "silu":
        raise ValueError("Llama loader requires SiLU")
    head_dim = hf.get("head_dim", hf["hidden_size"] // hf["num_attention_heads"])
    return {
        "num_layers": hf["num_hidden_layers"],
        "vocab_size": hf["vocab_size"],
        "embed_dim": hf["hidden_size"],
        "hidden_dim": hf["intermediate_size"],
        "num_heads": hf["num_attention_heads"],
        "head_dim": head_dim,
        "num_kv_heads": hf["num_key_value_heads"],
        "rope_theta": hf["rope_theta"],
        "norm_eps": hf["rms_norm_eps"],
        "use_tied_embedding": hf.get("tie_word_embeddings", False),
    }


def create_llama_model(model_config: dict[str, Any], mesh: Any) -> tuple[Any, str]:
    """Create a full-fine-tune Llama model from local safetensors only."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from tunix.models import safetensors_loader
    from tunix.models.qwen2 import model as qwen2
    from tunix.models.qwen2 import params

    if model_config.get("model_source") != "local":
        raise ValueError("Llama checkpoints must be staged locally")
    if model_config.get("lora_config"):
        raise ValueError("The local Llama path currently supports full fine-tuning")
    path = Path(model_config["model_path"]).expanduser().resolve()
    dimensions = llama_dimensions(json.loads((path / "config.json").read_text()))
    if (
        model_config.get("rope_theta", dimensions["rope_theta"])
        != dimensions["rope_theta"]
    ):
        raise ValueError("Llama recipe RoPE must match the staged config.json")
    config = qwen2.ModelConfig(
        **dimensions,
        dtype=getattr(jnp, model_config.get("dtype", "bfloat16")),
        remat_config=getattr(
            qwen2.RematConfig, model_config.get("remat_config", "NONE")
        ),
        use_flash_attention=model_config.get("use_flash_attention", False),
        flash_attention_block_size=model_config.get("flash_attention_block_size", 1024),
    )

    class ZeroBias(nnx.Module):
        def astype(self, dtype):
            return jnp.asarray(0, dtype=dtype)

    class Llama(qwen2.Qwen2):
        def __init__(self, config, *, rngs):
            super().__init__(config, rngs=rngs)
            for layer in self.layers:
                layer.attn.q_bias = ZeroBias()
                layer.attn.k_bias = ZeroBias()
                layer.attn.v_bias = ZeroBias()

    with jax.set_mesh(mesh):
        model = safetensors_loader.load_and_create_model(
            file_dir=str(path),
            model_class=Llama,
            config=config,
            key_mapping=params._get_key_and_transform_mapping,
            mesh=mesh,
            preprocess_fn=None,
            dtype=getattr(jnp, model_config.get("load_dtype", "bfloat16")),
        )
    return model, str(path)
