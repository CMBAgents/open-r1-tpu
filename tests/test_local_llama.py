import numpy as np
import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.model.export import llama_safetensors_entry, safetensors_entry_fn
from open_r1_tpu.model.llama import llama_dimensions


def test_llama_dimensions_come_from_checkpoint():
    hf = {
        "model_type": "llama",
        "num_hidden_layers": 12,
        "vocab_size": 32000,
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "rope_theta": 50000,
        "rms_norm_eps": 1e-5,
    }
    actual = llama_dimensions(hf)
    assert actual["head_dim"] == 128
    assert actual["rope_theta"] == 50000
    assert actual["num_layers"] == 12
    assert not actual["use_tied_embedding"]
    for key, value in [
        ("rope_scaling", {"factor": 2}),
        ("attention_bias", True),
        ("mlp_bias", True),
        ("hidden_act", "gelu"),
    ]:
        with pytest.raises(ValueError):
            llama_dimensions(hf | {key: value})


def test_llama_export_rejects_non_llama_parameters():
    assert safetensors_entry_fn("rowanai", "llama") is llama_safetensors_entry
    for path in ("layers.0.attn.q_bias", "layers.0.attn.q_norm.w"):
        with pytest.raises(ValueError):
            llama_safetensors_entry(path, np.zeros(4))


def test_comparison_matches_data_and_update_budget():
    a = load_config("recipes/rowanai/sft/config_rowanai.yaml")
    b = load_config("recipes/rowanai/sft/config_qwen.yaml")
    assert a["dataset"] == b["dataset"]
    assert a["optimizer"] == b["optimizer"]
    assert a["dataset"]["packing"] is False
    assert a["dataset"]["require_reasoning_tags"] is False
    assert a["training"]["max_steps"] == b["training"]["max_steps"] == 709
    assert (
        a["training"]["gradient_accumulation_steps"]
        == b["training"]["gradient_accumulation_steps"]
        == 8
    )
    assert a["training"]["checkpoint_dir"] != b["training"]["checkpoint_dir"]
    assert a["export"]["output_dir"] != b["export"]["output_dir"]
    assert a["model"]["architecture"] == "llama"
    assert a["model"]["rope_theta"] == 50000
    assert b["model"]["rope_theta"] == 300000


def _write_tiny_llama(path):
    """A random 2-layer Llama checkpoint in Hugging Face layout."""
    import json

    from safetensors.numpy import save_file

    hidden, inter, heads, kv_heads, vocab, layers = 64, 128, 4, 2, 128, 2
    head_dim = hidden // heads
    config = {
        "model_type": "llama",
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "rope_theta": 50000,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
    }
    (path / "config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(0)

    def w(*shape):
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    tensors = {
        "model.embed_tokens.weight": w(vocab, hidden),
        "model.norm.weight": np.ones(hidden, np.float32),
        "lm_head.weight": w(vocab, hidden),
    }
    for i in range(layers):
        pre = f"model.layers.{i}."
        tensors |= {
            pre + "self_attn.q_proj.weight": w(heads * head_dim, hidden),
            pre + "self_attn.k_proj.weight": w(kv_heads * head_dim, hidden),
            pre + "self_attn.v_proj.weight": w(kv_heads * head_dim, hidden),
            pre + "self_attn.o_proj.weight": w(hidden, heads * head_dim),
            pre + "mlp.gate_proj.weight": w(inter, hidden),
            pre + "mlp.up_proj.weight": w(inter, hidden),
            pre + "mlp.down_proj.weight": w(hidden, inter),
            pre + "input_layernorm.weight": np.ones(hidden, np.float32),
            pre + "post_attention_layernorm.weight": np.ones(hidden, np.float32),
        }
    save_file(tensors, str(path / "model.safetensors"))


def _tiny_llama_config(path, lora=None, export=False):
    model = {
        "model_name": "rowanai",
        "model_source": "local",
        "model_path": str(path),
        "architecture": "llama",
        "dtype": "float32",
        "load_dtype": "float32",
        "rng_seed": 0,
        "mesh": {"shape": [1, 1], "axis_names": ["fsdp", "tp"]},
    }
    if lora:
        model["lora_config"] = lora
    return {"model": model, "tokenizer": {}, "export": {"enabled": export}}


def test_local_llama_loads_with_and_without_lora(tmp_path):
    pytest.importorskip("tunix")
    pytest.importorskip("safetensors")
    import jax
    import jax.numpy as jnp
    from tunix.sft import utils as sft_utils
    from tunix.utils import mesh as mesh_utils

    from open_r1_tpu.model.loading import create_model

    _write_tiny_llama(tmp_path)
    mesh = mesh_utils.create_mesh((1, 1), ("fsdp", "tp"))
    lora = {
        "module_path": ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
        ".*gate_proj|.*up_proj|.*down_proj",
        "rank": 4,
        "alpha": 4.0,
    }

    base, _ = create_model(_tiny_llama_config(tmp_path), mesh)
    adapted, _ = create_model(_tiny_llama_config(tmp_path, lora), mesh)
    assert not sft_utils.is_lora_enabled(base)
    assert sft_utils.is_lora_enabled(adapted)

    tokens = jnp.array([[1, 5, 9, 17]], dtype=jnp.int32)
    positions = jnp.arange(4, dtype=jnp.int32)[None, :]
    mask = jnp.tril(jnp.ones((1, 4, 4), dtype=jnp.bool_))
    with jax.set_mesh(mesh):
        base_logits, _ = base(tokens, positions, None, mask)
        adapted_logits, _ = adapted(tokens, positions, None, mask)
    assert adapted_logits.shape == (1, 4, 128)
    # A fresh LoRA adapter starts as a no-op (B is zero-initialised).
    np.testing.assert_allclose(
        np.asarray(adapted_logits), np.asarray(base_logits), atol=1e-4
    )


def test_local_llama_lora_with_export_fails_before_training(tmp_path):
    pytest.importorskip("tunix")
    pytest.importorskip("safetensors")
    from tunix.utils import mesh as mesh_utils

    from open_r1_tpu.model.loading import create_model

    _write_tiny_llama(tmp_path)
    mesh = mesh_utils.create_mesh((1, 1), ("fsdp", "tp"))
    config = _tiny_llama_config(
        tmp_path, {"module_path": ".*q_proj", "rank": 4, "alpha": 4.0}, export=True
    )
    with pytest.raises(NotImplementedError, match=r"export\.enabled=false"):
        create_model(config, mesh)
