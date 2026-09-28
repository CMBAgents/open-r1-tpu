"""Merged-LoRA export for Qwen2, which the pinned Tunix does not ship.

The key-mapping tests run anywhere. The round trip needs Tunix and the staged
Qwen2.5-1.5B base (``models/Qwen2.5-1.5B``), so it runs on the TPU VM and
skips elsewhere: Tunix builds Qwen2 at the registered size for the model name,
so there is no tiny stand-in to use instead.
"""

from pathlib import Path

import numpy as np
import pytest

from open_r1_tpu.model.export import (
    _QWEN_LORA_TRANSPOSE_RULES,
    QWEN_LORA_MODULES,
    qwen_lora_state_key,
)

QWEN2_BASE = Path(__file__).parents[1] / "models/Qwen2.5-1.5B"


@pytest.mark.parametrize(
    ("lora_path", "key"),
    [
        ("layers.0.attn.q_proj", "model.layers.0.self_attn.q_proj.weight"),
        ("layers.27.attn.o_proj", "model.layers.27.self_attn.o_proj.weight"),
        ("layers.3.mlp.gate_proj", "model.layers.3.mlp.gate_proj.weight"),
        ("layers.3.mlp.down_proj", "model.layers.3.mlp.down_proj.weight"),
    ],
)
def test_qwen_lora_state_key(lora_path, key):
    assert qwen_lora_state_key(lora_path) == key


def test_every_lora_module_has_a_transpose_rule():
    assert set(_QWEN_LORA_TRANSPOSE_RULES) == set(QWEN_LORA_MODULES)


def _config(model_path, lora=None):
    model = {
        "model_name": "qwen2.5-1.5b",
        "model_id": "Qwen/Qwen2.5-1.5B",
        "model_source": "local",
        "model_path": str(model_path),
        # float32: Qwen2.5-1.5B's attention biases overflow bfloat16 inference.
        "dtype": "float32",
        "load_dtype": "float32",
        "rope_theta": 1000000,
        "rng_seed": 0,
        "mesh": {"shape": [1, 1], "axis_names": ["fsdp", "tp"]},
    }
    if lora:
        model["lora_config"] = lora
    return {"model": model, "tokenizer": {}, "export": {"enabled": False}}


def test_qwen2_lora_merged_export_reproduces_the_adapted_model(tmp_path):
    pytest.importorskip("tunix")
    if not (QWEN2_BASE / "model.safetensors").is_file():
        pytest.skip(f"needs the staged Qwen2.5-1.5B base at {QWEN2_BASE}")
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from tunix.utils import mesh as mesh_utils

    from open_r1_tpu.model.export import save_qwen2_lora_merged_model_as_safetensors
    from open_r1_tpu.model.loading import create_model

    mesh = mesh_utils.create_mesh((1, 1), ("fsdp", "tp"))
    module_path = "|".join(f".*{name}" for name in QWEN_LORA_MODULES)
    lora = {"module_path": module_path, "rank": 4, "alpha": 8.0}
    adapted, _ = create_model(_config(QWEN2_BASE, lora), mesh)

    # A fresh adapter is a no-op (B starts at zero), which would let a wrong
    # merge pass. Random B makes every adapted projection carry a real delta.
    rng = np.random.default_rng(0)
    touched = 0
    for path, value in nnx.iter_graph(adapted):
        if isinstance(value, nnx.LoRAParam) and "lora_b" in str(path[-1]):
            shape = value.value.shape
            value.value = jnp.asarray(
                rng.standard_normal(shape) * 0.02, dtype=value.value.dtype
            )
            touched += 1
    assert touched == 28 * len(QWEN_LORA_MODULES)

    merged_dir = tmp_path / "merged"
    save_qwen2_lora_merged_model_as_safetensors(
        local_model_path=str(QWEN2_BASE),
        output_dir=str(merged_dir),
        lora_model=adapted,
        rank=lora["rank"],
        alpha=lora["alpha"],
    )
    merged, _ = create_model(_config(merged_dir), mesh)
    base, _ = create_model(_config(QWEN2_BASE), mesh)

    tokens = jnp.array([[785, 3974, 13876, 38835, 34208, 916, 279, 15678]], jnp.int32)
    positions = jnp.arange(tokens.shape[1], dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_))
    with jax.set_mesh(mesh):
        adapted_logits = np.asarray(adapted(tokens, positions, None, mask)[0])
        merged_logits = np.asarray(merged(tokens, positions, None, mask)[0])
        base_logits = np.asarray(base(tokens, positions, None, mask)[0])

    # The adapter moved the model, and the merged checkpoint moved it the same
    # way. The base checkpoint is bfloat16, so the merge rounds each weight
    # once; the tolerance is a small fraction of the adapter's own effect.
    effect = np.abs(adapted_logits - base_logits).max()
    assert effect > 1.0
    np.testing.assert_allclose(merged_logits, adapted_logits, atol=0.05 * effect)
