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
