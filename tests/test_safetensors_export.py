import numpy as np
import pytest

from open_r1_tpu.safetensors_export import (
    collect_safetensors_state,
    qwen3_safetensors_entry,
)

EMBED, HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB = 8, 4, 2, 3, 10, 16


def _loader_transform(hf_tensor, permute, reshape):
    """Apply the loader's HF -> Tunix transform from qwen3/params.py."""
    value = hf_tensor
    if permute is not None:
        value = value.transpose(permute)
    if reshape is not None:
        value = value.reshape(reshape)
    return value


@pytest.mark.parametrize(
    "hf_key,path,hf_shape,permute,reshape",
    [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "layers.0.attn.q_proj.w",
            (HEADS * HEAD_DIM, EMBED),
            (1, 0),
            (EMBED, HEADS, HEAD_DIM),
        ),
        (
            "model.layers.3.self_attn.k_proj.weight",
            "layers.3.attn.k_proj.w",
            (KV_HEADS * HEAD_DIM, EMBED),
            (1, 0),
            (EMBED, KV_HEADS, HEAD_DIM),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            "layers.0.attn.o_proj.w",
            (EMBED, HEADS * HEAD_DIM),
            (1, 0),
            (HEADS, HEAD_DIM, EMBED),
        ),
        (
            "model.layers.1.mlp.gate_proj.weight",
            "layers.1.mlp.gate_proj.kernel",
            (INTER, EMBED),
            (1, 0),
            None,
        ),
        (
            "model.layers.1.mlp.down_proj.weight",
            "layers.1.mlp.down_proj.kernel",
            (EMBED, INTER),
            (1, 0),
            None,
        ),
        ("model.embed_tokens.weight", "embedder.input_embedding", (VOCAB, EMBED), None, None),
        ("model.norm.weight", "final_norm.w", (EMBED,), None, None),
        (
            "model.layers.2.self_attn.q_norm.weight",
            "layers.2.attn.q_norm.w",
            (HEAD_DIM,),
            None,
            None,
        ),
        (
            "model.layers.2.input_layernorm.weight",
            "layers.2.input_layernorm.w",
            (EMBED,),
            None,
            None,
        ),
        ("lm_head.weight", "lm_head.w", (VOCAB, EMBED), (1, 0), None),
    ],
)
def test_export_inverts_the_loader_transform(hf_key, path, hf_shape, permute, reshape):
    hf_tensor = np.arange(np.prod(hf_shape), dtype=np.float32).reshape(hf_shape)
    live = _loader_transform(hf_tensor, permute, reshape)

    key, exported = qwen3_safetensors_entry(path, live)

    assert key == hf_key
    np.testing.assert_array_equal(exported, hf_tensor)


def test_unknown_parameter_path_fails_loudly():
    with pytest.raises(ValueError, match="No safetensors mapping"):
        qwen3_safetensors_entry("layers.0.attn.rotary.w", np.zeros((2, 2)))


def test_flat_projection_is_rejected():
    with pytest.raises(ValueError, match="expected a 3D tensor"):
        qwen3_safetensors_entry("layers.0.attn.q_proj.w", np.zeros((4, 4)))


def test_duplicate_keys_are_rejected():
    params = [
        ("final_norm.w", np.zeros((2,))),
        ("final_norm.w", np.zeros((2,))),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        collect_safetensors_state(params)


def test_collected_state_uses_hf_names():
    state = collect_safetensors_state(
        [
            ("embedder.input_embedding", np.zeros((VOCAB, EMBED))),
            ("final_norm.w", np.zeros((EMBED,))),
        ]
    )
    assert set(state) == {"model.embed_tokens.weight", "model.norm.weight"}
