import importlib.util
from collections import UserDict
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "chat_qwen_tpu.py"
SPEC = importlib.util.spec_from_file_location("chat_qwen_tpu", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)


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
