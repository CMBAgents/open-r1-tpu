import json
import sys
import types

import numpy as np
import pytest

from open_r1_tpu.training.export import (
    SAFETENSORS_ENTRY_FNS,
    collect_safetensors_state,
    qwen2_safetensors_entry,
    qwen3_safetensors_entry,
    safetensors_entry_fn,
    write_turn_end_generation_config,
)

EMBED, HEADS, KV_HEADS, HEAD_DIM, INTER, VOCAB = 8, 4, 2, 3, 10, 16

ENTRY_FNS = [qwen2_safetensors_entry, qwen3_safetensors_entry]
ENTRY_FN_IDS = ["qwen2", "qwen3"]


def _loader_transform(hf_tensor, permute, reshape):
    """Apply the loader's HF -> Tunix transform from <family>/params.py."""
    value = hf_tensor
    if permute is not None:
        value = value.transpose(permute)
    if reshape is not None:
        value = value.reshape(reshape)
    return value


@pytest.mark.parametrize("entry_fn", ENTRY_FNS, ids=ENTRY_FN_IDS)
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
        (
            "model.embed_tokens.weight",
            "embedder.input_embedding",
            (VOCAB, EMBED),
            None,
            None,
        ),
        ("model.norm.weight", "final_norm.w", (EMBED,), None, None),
        (
            "model.layers.2.input_layernorm.weight",
            "layers.2.input_layernorm.w",
            (EMBED,),
            None,
            None,
        ),
        (
            "model.layers.2.post_attention_layernorm.weight",
            "layers.2.post_attention_layernorm.w",
            (EMBED,),
            None,
            None,
        ),
        # Only reached on an untied model: DeepSeek-R1-Distill-Qwen-1.5B has an
        # lm_head, Qwen2.5-Math-1.5B and the Qwen3 Base models do not.
        ("lm_head.weight", "lm_head.w", (VOCAB, EMBED), (1, 0), None),
    ],
)
def test_export_inverts_the_loader_transform(
    entry_fn, hf_key, path, hf_shape, permute, reshape
):
    """Every rule the two families share must invert identically in both."""
    hf_tensor = np.arange(np.prod(hf_shape), dtype=np.float32).reshape(hf_shape)
    live = _loader_transform(hf_tensor, permute, reshape)

    key, exported = entry_fn(path, live)

    assert key == hf_key
    np.testing.assert_array_equal(exported, hf_tensor)


@pytest.mark.parametrize(
    "hf_key,path,hf_shape",
    [
        ("model.layers.2.self_attn.q_norm.weight", "layers.2.attn.q_norm.w", HEAD_DIM),
        ("model.layers.0.self_attn.k_norm.weight", "layers.0.attn.k_norm.w", HEAD_DIM),
    ],
)
def test_qwen3_query_and_key_norms(hf_key, path, hf_shape):
    hf_tensor = np.arange(hf_shape, dtype=np.float32)
    key, exported = qwen3_safetensors_entry(path, hf_tensor)
    assert key == hf_key
    np.testing.assert_array_equal(exported, hf_tensor)


@pytest.mark.parametrize(
    "hf_key,path,size",
    [
        (
            "model.layers.0.self_attn.q_proj.bias",
            "layers.0.attn.q_bias",
            HEADS * HEAD_DIM,
        ),
        (
            "model.layers.3.self_attn.k_proj.bias",
            "layers.3.attn.k_bias",
            KV_HEADS * HEAD_DIM,
        ),
        (
            "model.layers.3.self_attn.v_proj.bias",
            "layers.3.attn.v_bias",
            KV_HEADS * HEAD_DIM,
        ),
    ],
)
def test_qwen2_projection_biases_pass_through_flat(hf_key, path, size):
    """The loader applies no transform to these, so neither does the export."""
    hf_tensor = np.arange(size, dtype=np.float32)
    key, exported = qwen2_safetensors_entry(path, hf_tensor)
    assert key == hf_key
    np.testing.assert_array_equal(exported, hf_tensor)


def test_families_reject_each_others_parameters():
    """A Qwen2 bias in a Qwen3 export means the wrong mapping was chosen."""
    with pytest.raises(ValueError, match="No safetensors mapping for Qwen3"):
        qwen3_safetensors_entry("layers.0.attn.q_bias", np.zeros((4,)))
    with pytest.raises(ValueError, match="No safetensors mapping for Qwen2"):
        qwen2_safetensors_entry("layers.0.attn.q_norm.w", np.zeros((4,)))


@pytest.mark.parametrize("entry_fn", ENTRY_FNS, ids=ENTRY_FN_IDS)
def test_unknown_parameter_path_fails_loudly(entry_fn):
    with pytest.raises(ValueError, match="No safetensors mapping"):
        entry_fn("layers.0.attn.rotary.w", np.zeros((2, 2)))


@pytest.mark.parametrize("entry_fn", ENTRY_FNS, ids=ENTRY_FN_IDS)
def test_flat_projection_is_rejected(entry_fn):
    with pytest.raises(ValueError, match="expected a 3D tensor"):
        entry_fn("layers.0.attn.q_proj.w", np.zeros((4, 4)))


def test_duplicate_keys_are_rejected():
    params = [
        ("final_norm.w", np.zeros((2,))),
        ("final_norm.w", np.zeros((2,))),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        collect_safetensors_state(params, qwen3_safetensors_entry)


def test_collected_state_uses_hf_names():
    state = collect_safetensors_state(
        [
            ("embedder.input_embedding", np.zeros((VOCAB, EMBED))),
            ("final_norm.w", np.zeros((EMBED,))),
        ],
        qwen2_safetensors_entry,
    )
    assert set(state) == {"model.embed_tokens.weight", "model.norm.weight"}


def _stub_tunix_registry(monkeypatch, module_name):
    """Stand in for tunix.models.automodel, which is TPU-only."""
    automodel = types.ModuleType("tunix.models.automodel")

    class ModelModule:
        MODEL = "model"
        PARAMS = "params"

    def get_model_module(name, which):
        assert which is ModelModule.MODEL
        return types.ModuleType(module_name)

    models = types.ModuleType("tunix.models")
    tunix = types.ModuleType("tunix")
    # setattr, as in tests/conftest.py: a bare ModuleType declares none of
    # these attributes, so assigning them directly is a type error.
    setattr(automodel, "ModelModule", ModelModule)  # noqa: B010
    setattr(automodel, "get_model_module", get_model_module)  # noqa: B010
    setattr(models, "automodel", automodel)  # noqa: B010
    setattr(tunix, "models", models)  # noqa: B010
    for name, module in (
        ("tunix", tunix),
        ("tunix.models", models),
        ("tunix.models.automodel", automodel),
    ):
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize(
    "module_name,expected",
    [
        ("tunix.models.qwen2.model", qwen2_safetensors_entry),
        ("tunix.models.qwen3.model", qwen3_safetensors_entry),
    ],
)
def test_entry_fn_follows_the_architecture_tunix_loaded(
    monkeypatch, module_name, expected
):
    """deepseek-r1-distill-qwen-1.5b is Qwen2; the name alone does not say so."""
    _stub_tunix_registry(monkeypatch, module_name)
    assert safetensors_entry_fn("deepseek-r1-distill-qwen-1.5b") is expected


def test_unsupported_architecture_names_what_is_implemented(monkeypatch):
    _stub_tunix_registry(monkeypatch, "tunix.models.gemma3.model")
    with pytest.raises(NotImplementedError) as excinfo:
        safetensors_entry_fn("gemma-3-1b-pt")
    message = str(excinfo.value)
    assert "gemma-3-1b-pt" in message
    for family in SAFETENSORS_ENTRY_FNS:
        assert family in message


class ChatTokenizer:
    """Renders <role>content</role> turns; every assistant turn ends </assistant>."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(character) for character in rendered]


TURN_END = ord("<")  # first token of this template's closing sequence "</assistant>"


def test_turn_end_token_is_prepended_to_copied_eos(tmp_path):
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps({"eos_token_id": 3, "do_sample": False}))
    write_turn_end_generation_config(
        output_dir=str(tmp_path), tokenizer=ChatTokenizer()
    )
    written = json.loads(path.read_text())
    assert written["eos_token_id"] == [TURN_END, 3]
    assert written["do_sample"] is False


def test_turn_end_token_is_not_duplicated(tmp_path):
    path = tmp_path / "generation_config.json"
    path.write_text(json.dumps({"eos_token_id": [TURN_END, 3]}))
    write_turn_end_generation_config(
        output_dir=str(tmp_path), tokenizer=ChatTokenizer()
    )
    assert json.loads(path.read_text())["eos_token_id"] == [TURN_END, 3]


def test_missing_generation_config_is_created(tmp_path):
    write_turn_end_generation_config(
        output_dir=str(tmp_path), tokenizer=ChatTokenizer()
    )
    written = json.loads((tmp_path / "generation_config.json").read_text())
    assert written == {"eos_token_id": [TURN_END]}
