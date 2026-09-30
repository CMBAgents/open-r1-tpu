"""Metrics and W&B logging for both training stages: backend options, metric
filtering and renaming, per-step rows, GRPO behaviour metrics and the eval
rollout table. No Tunix, JAX or W&B needed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.grpo.behaviour import (
    build_behaviour_metric_fn,
    completion_behaviour,
    group_signal,
)
from open_r1_tpu.model.metrics import (
    STEP_METRIC,
    StepAxisWandbBackend,
    SteppedTrainingMetricsBackend,
    metrics_logger_options,
    wandb_backend_kwargs,
)
from open_r1_tpu.sft.config import validate_sft_config

SFT_RECIPE = (
    Path(__file__).parents[1] / "recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml"
)


def load_sft(overrides=()):
    return load_config(SFT_RECIPE, list(overrides), validator=validate_sft_config)


def test_wandb_can_be_disabled_for_local_runs():
    config = load_sft(["training.wandb.enabled=false"])
    assert config["training"]["wandb"]["enabled"] is False
    assert wandb_backend_kwargs(config) == {"mode": "disabled"}


def test_wandb_backend_receives_run_metadata_and_resolved_config():
    config = load_sft(["training.wandb.entity=my-team"])
    kwargs = wandb_backend_kwargs(config)

    assert kwargs["entity"] == "my-team"
    assert kwargs["group"] == "qwen2.5-math-1.5b-reasoning-distillation"
    assert kwargs["job_type"] == "sft"
    assert kwargs["tags"] == config["training"]["wandb"]["tags"]
    assert kwargs["dir"] == config["training"]["metrics_log_dir"]
    assert kwargs["config"] is config


def test_metrics_options_use_custom_backends():
    class Options:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Backend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeMetricsLogger:
        MetricsLoggerOptions = Options
        TensorboardBackend = Backend
        WandbBackend = Backend

    config = load_sft()
    options = metrics_logger_options(config, FakeMetricsLogger)
    factories = options.kwargs["backend_kwargs"]["custom_backend"]

    assert len(factories) == 2
    assert factories[0]().kwargs == {
        "log_dir": config["training"]["metrics_log_dir"],
        "flush_every_n_steps": config["training"]["flush_every_n_steps"],
    }
    assert isinstance(factories[1](), SteppedTrainingMetricsBackend)

    disabled_config = load_sft(["training.wandb.enabled=false"])
    disabled_options = metrics_logger_options(disabled_config, FakeMetricsLogger)
    disabled_factories = disabled_options.kwargs["backend_kwargs"]["custom_backend"]
    assert len(disabled_factories) == 1


class Recorder:
    def __init__(self):
        self.calls = []
        self.closed = False

    def log_scalar(self, event, value, **kwargs):
        self.calls.append((event, value, kwargs.get("step")))

    def close(self):
        self.closed = True


def test_sft_metrics_pass_and_unstepped_or_system_events_are_dropped():
    inner = Recorder()
    backend = SteppedTrainingMetricsBackend(inner)

    backend.log_scalar("/train/loss", 1.25, step=3)
    backend.log_scalar("/eval/loss", 1.5, step=3)
    backend.log_scalar("/jax/orbax/write/gbytes_per_sec", 10.0)
    backend.log_scalar("/jax/core/compile/backend_compile_duration", 12.0)
    backend.log_scalar("/train/perplexity", 3.5)
    backend.close()

    assert inner.calls == [("/train/loss", 1.25, 3), ("/eval/loss", 1.5, 3)]
    assert inner.closed is True


def test_rl_metrics_are_renamed_by_mode_and_system_events_dropped():
    inner = Recorder()
    backend = SteppedTrainingMetricsBackend(inner)
    backend.log_scalar("rewards/train/mean", 0.5, step=3)
    backend.log_scalar("/actor/train/kl", 0.01, step=3)
    backend.log_scalar("completions/eval/mean_length", 120.0, step=3)
    backend.log_scalar("behaviour/train/answer_line_frac", 0.9, step=3)
    backend.log_scalar("/train/loss", 1.0, step=3)  # SFT naming, unchanged
    backend.log_scalar("jax/core/compile/backend_compile_duration", 5.0, step=3)
    backend.log_scalar("jax/orbax/write/gbytes", 1.0)
    backend.log_scalar("rewards/train/mean", 0.5)  # no step
    backend.log_scalar("rewards/other/mean", 0.5, step=3)  # unknown mode
    assert [event for event, _, _ in inner.calls] == [
        "/train/rewards/mean",
        "/train/actor/kl",
        "/eval/completions/mean_length",
        "/train/behaviour/answer_line_frac",
        "/train/loss",
    ]


class FakeWandb:
    def __init__(self):
        self.rows = []
        self.defined = []

    def define_metric(self, name, **kwargs):
        self.defined.append((name, kwargs))

    def log(self, row):
        self.rows.append(row)


def test_step_axis_backend_writes_one_row_per_step():
    wandb = FakeWandb()
    inner = SimpleNamespace(wandb=wandb, _is_active=True, closed=False)
    inner.close = lambda: setattr(inner, "closed", True)
    backend = StepAxisWandbBackend(inner)
    backend.log_scalar("/train/actor/loss", np.float32(1.5), step=1)
    backend.log_scalar("/train/rewards/mean", np.array([0.0, 1.0]), step=1)
    backend.log_scalar("/train/actor/loss", 1.25, step=2)
    backend.log_scalar("/eval/rewards/mean", 0.25, step=1)  # late arrival
    backend.log_scalar("/train/actor/loss", 9.0)  # no step: ignored
    backend.close()
    assert wandb.rows == [
        {"train/actor/loss": 1.5, "train/rewards/mean": 0.5, STEP_METRIC: 1},
        {"train/actor/loss": 1.25, STEP_METRIC: 2},
        {"eval/rewards/mean": 0.25, STEP_METRIC: 1},
    ]
    assert wandb.defined == [
        (STEP_METRIC, {}),
        ("*", {"step_metric": STEP_METRIC}),
    ]
    assert inner.closed


def test_step_axis_backend_is_silent_without_an_active_run():
    wandb = FakeWandb()
    inner = SimpleNamespace(wandb=wandb, _is_active=False, close=lambda: None)
    backend = StepAxisWandbBackend(inner)
    backend.log_scalar("/train/loss", 1.0, step=1)
    backend.close()
    assert wandb.rows == []


def test_completion_behaviour_reads_shape_and_loops():
    completions = [
        "3 x 4 = 12.\nAnswer: 12",
        "so \\boxed{7}",
        "loop\nloop\nloop\nno number",
        "Answer: 5",
    ]
    m = completion_behaviour(completions)
    assert m["behaviour/answer_line_frac"] == 0.5
    assert m["behaviour/boxed_frac"] == 0.25
    assert m["behaviour/number_found_frac"] == 0.75
    assert m["behaviour/chars_mean"] == sum(len(c) for c in completions) / 4
    assert m["behaviour/repeated_line_frac"] == 0.25
    assert 0 < m["behaviour/distinct_4gram_mean"] <= 1


def test_group_signal_counts_groups_that_can_teach():
    rewards = [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]  # three groups of 4
    m = group_signal(rewards, 4)
    assert m["signal/groups_any_reward_frac"] == pytest.approx(2 / 3)
    assert m["signal/groups_mixed_frac"] == pytest.approx(1 / 3)
    assert m["signal/groups_all_reward_frac"] == pytest.approx(1 / 3)
    assert m["signal/correct_per_group_mean"] == pytest.approx(5 / 3)
    assert group_signal([1, 0, 1], 2) == {}


def test_behaviour_metric_fn_returns_tunix_metric_tuples():
    fn = build_behaviour_metric_fn(2)
    out = fn(
        prompts=["p", "p"],
        completions=["Answer: 1", "x"],
        rewards=np.array([1.0, 0.0]),
        advantages=np.array([0.7, -0.7]),
        answer=np.array(["1", "1"]),
    )
    value, op = out["signal/groups_mixed_frac"]
    assert value == 1.0 and op is np.mean
    assert all(isinstance(v, tuple) and len(v) == 2 for v in out.values())


def test_eval_table_logger_logs_one_table_per_eval_step(monkeypatch):
    logged = []

    class Table:
        def __init__(self, columns, data):
            self.columns, self.data = columns, data

    run = SimpleNamespace(log=lambda row: logged.append(row))
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=run, Table=Table))
    from open_r1_tpu.grpo.run import build_eval_table_logger

    add, flush = build_eval_table_logger(max_chars=5)
    cols = {"question": np.array(["q1", "q1"]), "answer": np.array(["4", "4"])}
    add(0, ["Answer: 4", "nope"], [1.0, 0.0], **cols)
    add(0, ["Answer: 4", "nope"], [1.0, 0.0], **cols)
    add(100, ["x", "y"], [0.0, 0.0], **cols)
    assert len(logged) == 1
    first = logged[0]
    assert first["global_step"] == 0
    assert first["eval/rollouts"].columns == [
        "step",
        "question",
        "answer",
        "completion",
        "reward",
    ]
    assert first["eval/rollouts"].data[0] == [0, "q1", "4", "Answe", 1.0]
    assert len(first["eval/rollouts"].data) == 4
    flush()
    assert len(logged) == 2 and logged[1]["global_step"] == 100
    flush()
    assert len(logged) == 2


def test_eval_table_logger_is_a_no_op_without_a_run(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(run=None, Table=None))
    from open_r1_tpu.grpo.run import build_eval_table_logger

    add, flush = build_eval_table_logger()
    add(0, ["a"], [0.0])
    flush()
