import importlib.util
import sys
from collections import UserDict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "chat_qwen_tpu.py"
SPEC = importlib.util.spec_from_file_location("chat_qwen_tpu", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)
sys.modules["chat_qwen_tpu"] = chat

COMPLETION_SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "complete_qwen_tpu.py"
COMPLETION_SPEC = importlib.util.spec_from_file_location(
    "complete_qwen_tpu", COMPLETION_SCRIPT_PATH
)
assert COMPLETION_SPEC is not None and COMPLETION_SPEC.loader is not None
completion = importlib.util.module_from_spec(COMPLETION_SPEC)
COMPLETION_SPEC.loader.exec_module(completion)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt) -> Any:
        assert add_generation_prompt is True
        rendered = "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        if tokenize:
            return list(range(len(rendered)))
        return rendered + "|assistant:"


class BatchedFakeTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        result = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return [result] if tokenize else result


class MappingFakeTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        result = super().apply_chat_template(messages, tokenize, add_generation_prompt)
        return UserDict({"input_ids": result}) if tokenize else result


def test_render_prompt_includes_system_history_and_generation_prefix():
    rendered = chat.render_prompt(
        FakeTokenizer(),
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        "Be concise.",
    )

    assert rendered == "system:Be concise.|user:Hello|assistant:Hi|assistant:"


def test_fit_history_discards_oldest_complete_turn():
    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "new"},
    ]

    fitted, count, removed_turns = chat.fit_history(
        FakeTokenizer(), history, "", max_prompt_length=8
    )

    assert fitted == [{"role": "user", "content": "new"}]
    assert count == len("user:new")
    assert removed_turns == 1


def test_fit_history_rejects_a_single_message_that_exceeds_the_budget():
    fitted, count, removed_turns = chat.fit_history(
        FakeTokenizer(),
        [{"role": "user", "content": "too long"}],
        "",
        max_prompt_length=2,
    )

    assert fitted is None
    assert count == len("user:too long")
    assert removed_turns == 0


def test_prompt_token_count_accepts_a_batch_of_one():
    assert chat.prompt_token_count(
        BatchedFakeTokenizer(), [{"role": "user", "content": "Hi"}], ""
    ) == len("user:Hi")


def test_prompt_token_count_accepts_a_batch_encoding_mapping():
    assert chat.prompt_token_count(
        MappingFakeTokenizer(), [{"role": "user", "content": "Hi"}], ""
    ) == len("user:Hi")


class FakeCompletionSampler:
    def __init__(self):
        self.kwargs: dict[str, Any] = {}

    def tokenize(self, prompt):
        return list(range(len(prompt)))

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(text=[" Paris."])


def test_raw_completion_passes_the_prompt_unchanged():
    sampler = FakeCompletionSampler()
    args = SimpleNamespace(max_new_tokens=100, max_prompt_length=2048, seed=42)

    generated = completion.generate_completion(
        sampler, "The capital of France is", args
    )

    assert generated == " Paris."
    assert sampler.kwargs["input_strings"] == ["The capital of France is"]
    assert sampler.kwargs["max_generation_steps"] == 100


def test_raw_completion_pads_the_sampler_batch_for_fsdp():
    sampler = FakeCompletionSampler()
    args = SimpleNamespace(
        max_new_tokens=100,
        max_prompt_length=2048,
        seed=42,
        sampler_fsdp_size=2,
    )

    completion.generate_completion(sampler, "The capital of France is", args)

    assert sampler.kwargs["input_strings"] == [
        "The capital of France is",
        "The capital of France is",
    ]


def test_completion_runtime_disables_flash_attention(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_load_runtime(args, *, use_flash_attention):
        captured["args"] = args
        captured["use_flash_attention"] = use_flash_attention
        return sentinel

    monkeypatch.setattr(completion, "load_runtime", fake_load_runtime)
    args = SimpleNamespace()

    assert completion.load_completion_runtime(args) is sentinel
    assert captured == {"args": args, "use_flash_attention": False}


def test_completion_prompt_length_need_not_match_splash_block(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors").touch()
    args = SimpleNamespace(
        model_path=str(model_path),
        max_new_tokens=100,
        max_prompt_length=17,
    )

    completion.validate_options(args)

    assert args.model_path == str(model_path.resolve())


class StopTokenTokenizer:
    IDS: ClassVar[dict[str, int]] = {"<|im_end|>": 151645, "<|endoftext|>": 151643}

    def convert_tokens_to_ids(self, token):
        return self.IDS.get(token)


class AdapterTokenizer:
    """Stands in for Tunix's adapter, which wraps the real tokenizer."""

    def __init__(self):
        self.tokenizer = StopTokenTokenizer()


def test_stop_tokens_lead_with_the_end_of_turn_marker():
    # Qwen3 ends turns with <|im_end|>, but the base tokenizer_config names
    # <|endoftext|> as EOS, which a chat model never emits.
    assert chat.stop_token_ids(StopTokenTokenizer()) == [151645, 151643]


def test_stop_tokens_are_found_through_the_tunix_adapter():
    assert chat.stop_token_ids(AdapterTokenizer()) == [151645, 151643]


def test_a_tokenizer_without_the_turn_markers_is_rejected():
    class Bare:
        def convert_tokens_to_ids(self, token):
            return None

    with pytest.raises(ValueError, match="tokenizer defines none"):
        chat.stop_token_ids(Bare())


def test_clean_reply_drops_only_the_trailing_turn_marker():
    assert chat.clean_reply("Paris.<|im_end|>") == "Paris."
    assert chat.clean_reply("  Paris.  ") == "Paris."


def test_visible_reply_hides_the_empty_reasoning_scaffold():
    # The template opens every assistant turn with this when the message
    # carries no trace, so a model trained on SmolTalk learns to emit it.
    assert chat.visible_reply("<think>\n\n</think>\n\nParis.<|im_end|>") == "Paris."


def test_visible_reply_keeps_a_reasoning_trace_that_has_content():
    reply = "<think>\nFrance's capital\n</think>\n\nParis."

    assert chat.visible_reply(reply) == reply


def test_model_config_carries_lora_only_when_asked():
    assert "lora_config" not in chat.model_config(
        "/models/base",
        0,
        use_flash_attention=False,
        model_name="qwen3-1.7b-base",
        mesh_shape=(1, 1),
    )
    lora = {"rank": 8, "alpha": 8.0, "module_path": ".*q_proj"}
    assert (
        chat.model_config(
            "/models/base",
            0,
            use_flash_attention=False,
            model_name="qwen3-1.7b-base",
            mesh_shape=(1, 1),
            lora_config=lora,
        )["lora_config"]
        == lora
    )


def test_model_config_carries_the_discovered_mesh_shape():
    assert chat.model_config(
        "/models/base",
        0,
        use_flash_attention=False,
        model_name="qwen3-1.7b-base",
        mesh_shape=(1, 4),
    )["mesh"] == {"shape": [1, 4], "axis_names": ["fsdp", "tp"]}


def test_model_settings_are_derived_from_the_local_config(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"_name_or_path":"Qwen/Qwen2.5-Math-1.5B","num_key_value_heads":2}',
        encoding="utf-8",
    )

    assert chat.model_settings_for_path(str(tmp_path)) == (
        "qwen2.5-math-1.5b",
        2,
    )


def test_model_settings_accept_an_explicit_name_when_source_metadata_is_missing(
    tmp_path,
):
    (tmp_path / "config.json").write_text('{"num_key_value_heads":2}', encoding="utf-8")

    assert chat.model_settings_for_path(str(tmp_path), "qwen2.5-math-1.5b") == (
        "qwen2.5-math-1.5b",
        2,
    )


def test_mesh_shape_uses_every_visible_tpu_device():
    devices = [SimpleNamespace(platform="tpu", id=device_id) for device_id in range(4)]

    assert chat.mesh_shape_for_devices(devices, num_kv_heads=8) == (1, 4)


def test_qwen2_5_mesh_uses_fully_sharded_data_parallelism_after_tp():
    devices = [SimpleNamespace(platform="tpu", id=device_id) for device_id in range(4)]

    assert chat.mesh_shape_for_devices(devices, num_kv_heads=2) == (2, 2)
    assert chat.pad_input_strings_for_fsdp(["prompt"], fsdp_size=2) == [
        "prompt",
        "prompt",
    ]


def test_mesh_shape_rejects_no_devices_or_non_tpu_devices():
    with pytest.raises(RuntimeError, match="at least one visible TPU"):
        chat.mesh_shape_for_devices([], num_kv_heads=8)
    with pytest.raises(RuntimeError, match="every visible JAX device"):
        chat.mesh_shape_for_devices(
            [SimpleNamespace(platform="cpu", id=0)], num_kv_heads=8
        )


def test_mesh_shape_rejects_a_tpu_count_that_cannot_tensor_parallelize_qwen3():
    devices = [SimpleNamespace(platform="tpu", id=device_id) for device_id in range(3)]

    with pytest.raises(RuntimeError, match="dividing its 8 KV heads"):
        chat.mesh_shape_for_devices(devices, num_kv_heads=8)


def test_load_runtime_wires_qwen2_5_into_the_four_chip_mesh(monkeypatch):
    captured: dict[str, Any] = {}
    mesh = object()
    tokenizer = object()
    sampler_instance = object()
    model = SimpleNamespace(
        config=SimpleNamespace(num_layers=28, num_kv_heads=2, head_dim=128)
    )

    def create_mesh(shape, axis_names):
        captured["mesh"] = (shape, axis_names)
        return mesh

    def create_model(config, received_mesh):
        captured["model"] = (config, received_mesh)
        return model, "tokenizer-path"

    def cache_config(**kwargs):
        captured["cache"] = kwargs
        return SimpleNamespace(**kwargs)

    def sampler(**kwargs):
        captured["sampler"] = kwargs
        return sampler_instance

    fake_jax: Any = ModuleType("jax")
    fake_jax.devices = lambda: [
        SimpleNamespace(platform="tpu", id=device_id) for device_id in range(4)
    ]
    fake_model_utils: Any = ModuleType("tunix.cli.utils.model")
    fake_model_utils.create_tokenizer = lambda _config, _path: tokenizer
    fake_sampler_lib: Any = ModuleType("tunix.generate.sampler")
    fake_sampler_lib.CacheConfig = cache_config
    fake_sampler_lib.Sampler = sampler
    fake_mesh_utils: Any = ModuleType("tunix.utils.mesh")
    fake_mesh_utils.create_mesh = create_mesh
    fake_training_run: Any = ModuleType("open_r1_tpu.training.run")
    fake_training_run._create_model = create_model
    monkeypatch.setattr(
        chat,
        "model_settings_for_path",
        lambda *_args: ("qwen2.5-math-1.5b", 2),
    )

    for name, module in {
        "jax": fake_jax,
        "tunix": ModuleType("tunix"),
        "tunix.cli": ModuleType("tunix.cli"),
        "tunix.cli.utils": ModuleType("tunix.cli.utils"),
        "tunix.cli.utils.model": fake_model_utils,
        "tunix.generate": ModuleType("tunix.generate"),
        "tunix.generate.sampler": fake_sampler_lib,
        "tunix.utils": ModuleType("tunix.utils"),
        "tunix.utils.mesh": fake_mesh_utils,
        "open_r1_tpu.training.run": fake_training_run,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    args = SimpleNamespace(
        model_path="/models/base",
        seed=42,
        checkpoint_dir=None,
        recipe=None,
        max_prompt_length=1024,
        max_new_tokens=128,
        model_name=None,
    )

    loaded_mesh, loaded_tokenizer, loaded_sampler, loaded_model = chat.load_runtime(
        args
    )

    assert (loaded_mesh, loaded_tokenizer, loaded_model) == (mesh, tokenizer, model)
    assert loaded_sampler is sampler_instance
    assert args.sampler_fsdp_size == 2
    assert captured["mesh"] == ((2, 2), ("fsdp", "tp"))
    assert captured["model"] == (
        {
            "model": {
                **chat.model_config(
                    "/models/base",
                    42,
                    use_flash_attention=False,
                    model_name="qwen2.5-math-1.5b",
                    mesh_shape=(2, 2),
                )
            },
            "tokenizer": chat.tokenizer_config("/models/base"),
        },
        mesh,
    )


def test_recipe_restore_settings_match_the_distill_recipe():
    recipe = (
        Path(__file__).parents[1]
        / "recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml"
    )
    lora_config, checkpoint_dir = chat.recipe_restore_settings(str(recipe))

    # Restoring under a different geometry than training wrote would produce
    # confident nonsense rather than an error, so these must come from there.
    assert lora_config["rank"] == 64
    assert lora_config["alpha"] == 64.0
    assert checkpoint_dir.endswith("OpenR1-Distill-Qwen3-1.7B/checkpoints")


def test_full_finetune_recipe_restores_all_parameters():
    # The instruct recipe trains all parameters, so its checkpoints carry the
    # full model state and restore must not be limited to LoRA adapters.
    recipe = (
        Path(__file__).parents[1]
        / "recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml"
    )
    lora_config, checkpoint_dir = chat.recipe_restore_settings(str(recipe))

    assert lora_config is None
    assert checkpoint_dir.endswith("Qwen3-1.7B-Instruct/checkpoints")


def test_chat_prompt_length_need_not_match_splash_block(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    args = SimpleNamespace(
        max_new_tokens=8,
        max_prompt_length=17,
        temperature=0.0,
        model_path=str(tmp_path),
        recipe=None,
        checkpoint_dir=None,
    )

    chat.validate_options(args)

    assert args.model_path == str(tmp_path.resolve())


def test_checkpoint_dir_without_a_recipe_is_rejected(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    args = SimpleNamespace(
        max_new_tokens=8,
        max_prompt_length=chat.FLASH_ATTENTION_BLOCK_SIZE,
        temperature=0.0,
        model_path=str(tmp_path),
        recipe=None,
        checkpoint_dir="artifacts/run/checkpoints",
    )

    with pytest.raises(ValueError, match="--checkpoint-dir needs --recipe"):
        chat.validate_options(args)


def _checkpoint_root(tmp_path, steps):
    for step in steps:
        (tmp_path / str(step)).mkdir()
    (tmp_path / "not-a-step").mkdir()
    return str(tmp_path)


def test_available_steps_ignores_non_step_directories(tmp_path):
    root = _checkpoint_root(tmp_path, [1500, 1000])

    assert chat.available_steps(root) == [1000, 1500]


def test_available_steps_tolerates_a_missing_root(tmp_path):
    assert chat.available_steps(str(tmp_path / "absent")) == []


def test_resolve_step_defaults_to_the_latest_written():
    assert chat.resolve_step("/absent", None) is None


def test_resolve_step_rejects_a_step_that_was_never_saved(tmp_path):
    # A run killed at 1744 saved 1500, and 1744 is the step its log reported.
    root = _checkpoint_root(tmp_path, [1000, 1500])

    with pytest.raises(FileNotFoundError, match="1000, 1500"):
        chat.resolve_step(root, 1744)


def test_resolve_step_accepts_a_step_that_was_saved(tmp_path):
    root = _checkpoint_root(tmp_path, [1000, 1500])

    assert chat.resolve_step(root, 1000) == 1000
