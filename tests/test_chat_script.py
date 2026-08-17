import importlib.util
import sys
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "chat_qwen_tpu.py"
SPEC = importlib.util.spec_from_file_location("chat_qwen_tpu", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)
sys.modules["chat_qwen_tpu"] = chat

COMPLETION_SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "complete_qwen_tpu.py"
)
COMPLETION_SPEC = importlib.util.spec_from_file_location(
    "complete_qwen_tpu", COMPLETION_SCRIPT_PATH
)
assert COMPLETION_SPEC is not None and COMPLETION_SPEC.loader is not None
completion = importlib.util.module_from_spec(COMPLETION_SPEC)
COMPLETION_SPEC.loader.exec_module(completion)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
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
        self.kwargs = None

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
