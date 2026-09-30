"""scripts/chat_tpu.py and scripts/complete_tpu.py, without a TPU. The model
loading they share is tested in test_model_checkpoint.py."""

import importlib.util
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"


def _load_script(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


chat = _load_script("chat_tpu")
completion = _load_script("complete_tpu")


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
    # Qwen3's template opens an assistant turn with this when the message
    # carries no trace, so a model trained on such data learns to emit it.
    assert chat.visible_reply("<think>\n\n</think>\n\nParis.<|im_end|>") == "Paris."


def test_visible_reply_keeps_a_reasoning_trace_that_has_content():
    reply = "<think>\nFrance's capital\n</think>\n\nParis."

    assert chat.visible_reply(reply) == reply


def test_colour_reasoning_dims_the_trace_through_its_closing_marker():
    assert chat.colour_reasoning("<think>\nsum digits\n</think>\n\n79.") == (
        f"{chat.REASONING_COLOUR}<think>\nsum digits\n</think>"
        f"{chat.COLOUR_RESET}\n\n79."
    )


def test_colour_reasoning_handles_a_trace_without_an_opening_marker():
    # ChatML does not pre-open the block, and <think> is ordinary tokens to
    # this tokeniser, so the model sometimes goes straight into the trace.
    assert chat.colour_reasoning("sum digits</think>79.") == (
        f"{chat.REASONING_COLOUR}sum digits</think>{chat.COLOUR_RESET}79."
    )


def test_colour_reasoning_dims_an_unclosed_trace_entirely():
    # A trace that ran out the token budget never closes; all of it is trace.
    assert chat.colour_reasoning("<think>endless deliberation") == (
        f"{chat.REASONING_COLOUR}<think>endless deliberation{chat.COLOUR_RESET}"
    )


def test_colour_reasoning_leaves_a_plain_reply_untouched():
    assert chat.colour_reasoning("Paris.") == "Paris."


def test_sampler_top_p_gates_the_stochastic_sampling_mode():
    # The pinned sampler greedy-decodes whenever top_p is None, silently
    # ignoring temperature and seed, so temperature 0 must map to None and
    # anything else must carry the top_p through.
    assert chat.sampler_top_p(0.0, 0.95) is None
    assert chat.sampler_top_p(0.6, 0.95) == 0.95


def test_validate_options_rejects_top_p_out_of_range(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    args = SimpleNamespace(
        max_new_tokens=8,
        max_prompt_length=17,
        temperature=0.6,
        top_p=0.0,
        model_path=str(tmp_path),
        recipe=None,
        checkpoint_dir=None,
    )

    with pytest.raises(ValueError, match="--top-p must be in"):
        chat.validate_options(args)


def test_chat_prompt_length_need_not_match_splash_block(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    args = SimpleNamespace(
        max_new_tokens=8,
        max_prompt_length=17,
        temperature=0.0,
        top_p=0.95,
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
        max_prompt_length=chat.DEFAULT_MAX_PROMPT_LENGTH,
        temperature=0.0,
        top_p=0.95,
        model_path=str(tmp_path),
        recipe=None,
        checkpoint_dir="artifacts/run/checkpoints",
    )

    with pytest.raises(ValueError, match="--checkpoint-dir needs --recipe"):
        chat.validate_options(args)


def test_a_recipe_defaults_the_model_path_to_the_base_it_trains_from(tmp_path):
    # Restoring onto any other base would serve the checkpoint at the wrong
    # RoPE theta for the RoPE-300k distillation recipe.
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"")
    recipe = tmp_path / "recipe.yaml"
    distill = (
        Path(__file__).parents[1]
        / "recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml"
    )
    recipe.write_text(f"extends: {distill}\nmodel:\n  model_path: {base}\n")
    args = SimpleNamespace(
        max_new_tokens=8,
        max_prompt_length=17,
        temperature=0.0,
        top_p=0.95,
        model_path=None,
        recipe=str(recipe),
        checkpoint_dir=None,
    )

    chat.validate_options(args)

    assert args.model_path == str(base.resolve())
