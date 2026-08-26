import argparse
import json

import pytest

from open_r1_tpu.evaluation import benchmark


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert not tokenize
        assert add_generation_prompt
        return "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )

    def encode(self, text):
        del text
        return [0, 1, 2]


def test_prompts_are_deterministic_distinct_and_include_the_system_prompt():
    questions = benchmark.benchmark_questions(4)
    prompts = benchmark.render_prompts(FakeTokenizer(), questions, "system")

    assert questions == benchmark.benchmark_questions(4)
    assert len(set(questions)) == 4
    assert all(prompt.startswith("system:system|") for prompt in prompts)
    assert benchmark.prompt_digest(prompts) == benchmark.prompt_digest(prompts)


def test_static_batches_reject_a_smaller_tail():
    with pytest.raises(ValueError, match="not divisible"):
        benchmark.split_batches(["a", "b", "c"], 2)


def test_vllm_payload_forces_a_fixed_length_greedy_decode():
    payload = benchmark.vllm_completion_payload(
        model_name="merged", prompt="prompt", max_new_tokens=128, seed=7
    )

    assert payload["temperature"] == 0.0
    assert payload["ignore_eos"] is True
    assert payload["max_tokens"] == 128
    assert payload["seed"] == 7


def test_workload_excludes_warmup_and_summarizes_exact_tokens(monkeypatch):
    times = iter(
        [
            0.0,
            3.0,  # warmup
            10.0,  # overall start
            10.0,
            12.0,  # timed batch 1
            12.0,
            14.0,  # timed batch 2
            14.0,  # overall end
        ]
    )
    monkeypatch.setattr(benchmark.time, "perf_counter", lambda: next(times))

    calls = []

    def run_batch(prompts, batch_size):
        calls.append(list(prompts))
        return benchmark.BatchOutput((8,) * batch_size)

    measurements = benchmark.run_workload(
        backend="fake",
        prompts=["a", "b", "c", "d"],
        prompt_tokens=[2, 2, 2, 2],
        batch_sizes=[2],
        repeats=1,
        max_new_tokens=8,
        run_batch=run_batch,
    )

    assert len(calls) == 3
    assert measurements[0]["warmup_seconds"] == 3.0
    assert measurements[0]["elapsed_seconds"] == 4.0
    assert measurements[0]["completion_tokens"] == 32
    assert measurements[0]["completion_tokens_per_second"] == 8.0


def _result(backend, rate_by_batch):
    return {
        "backend": backend,
        "model_path": "/tmp/model",
        "prompt_sha256": "abc",
        "config": {
            "batch_sizes": list(rate_by_batch),
            "prompt_count": 8,
            "repeats": 1,
            "max_new_tokens": 8,
            "max_prompt_length": 64,
            "temperature": 0.0,
            "fixed_length": True,
        },
        "runtime": {backend: "1.0"},
        "startup_seconds": 10.0,
        "measurements": [
            {
                "batch_size": batch_size,
                "completion_tokens_per_second": rate,
                "samples_per_second": rate / 8,
                "elapsed_seconds": 1.0,
            }
            for batch_size, rate in rate_by_batch.items()
        ],
    }


def test_comparison_matches_batch_sizes_and_reports_the_ratio():
    comparison = benchmark.build_comparison(
        _result("vllm", {1: 20.0, 8: 80.0}),
        _result("tunix", {1: 30.0, 8: 60.0}),
    )

    assert comparison["rows"][0]["tunix_over_vllm"] == 1.5
    assert comparison["rows"][1]["tunix_over_vllm"] == 0.75
    markdown = benchmark.comparison_markdown(comparison)
    assert "| 1 | 20.00 | 30.00 | 1.50x |" in markdown
    assert "current serial LightEval" in markdown


def test_comparison_rejects_different_prompt_sets():
    vllm = _result("vllm", {1: 20.0})
    tunix = _result("tunix", {1: 30.0})
    tunix["prompt_sha256"] = "different"

    with pytest.raises(ValueError, match="prompt_sha256"):
        benchmark.build_comparison(vllm, tunix)


def test_write_json_is_stable_and_readable(tmp_path):
    output = tmp_path / "result.json"
    benchmark.write_json(output, {"b": 2, "a": 1})

    assert benchmark.read_json(output) == {"a": 1, "b": 2}
    assert json.loads(output.read_text()) == {"a": 1, "b": 2}


@pytest.mark.parametrize("system_prompt", [None, "Recipe prompt, verbatim."])
def test_run_command_uses_the_recipes_system_prompt_verbatim(
    tmp_path, monkeypatch, system_prompt
):
    # No fallback: a recipe that deliberately sets no prompt (None) must be
    # benchmarked that way, and one that sets a prompt is benchmarked with
    # exactly that text -- neither gets substituted here.
    captured = {}

    monkeypatch.setattr(benchmark, "load_eval_config", lambda path: {})
    monkeypatch.setattr(
        benchmark,
        "resolve_eval_settings",
        lambda config: {
            "model_path": "m",
            "served_model_name": "m",
            "base_url": "http://x",
            "system_prompt": system_prompt,
        },
    )
    monkeypatch.setattr(benchmark, "_load_tokenizer", lambda path: FakeTokenizer())

    def fake_render_prompts(tokenizer, questions, prompt):
        captured["system_prompt"] = prompt
        return list(questions)

    monkeypatch.setattr(benchmark, "render_prompts", fake_render_prompts)
    monkeypatch.setattr(benchmark, "_run_vllm", lambda **kwargs: ([], {"vllm": None}))

    args = argparse.Namespace(
        prompt_count=1,
        max_new_tokens=8,
        max_prompt_length=64,
        batch_sizes=[1],
        eval_config="unused.yaml",
        model_path=None,
        backend="vllm",
        repeats=1,
        seed=0,
        startup_seconds=None,
        output=str(tmp_path / "out.json"),
    )

    benchmark._run_command(args)

    assert captured["system_prompt"] == system_prompt
