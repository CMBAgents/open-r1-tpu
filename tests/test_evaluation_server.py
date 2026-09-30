"""Tests for `open_r1_tpu.evaluation.server`: the vLLM command line and, with a
live server, its readiness.
"""

import json
import os
import urllib.request
from pathlib import Path

import pytest

from open_r1_tpu.evaluation import config as eval_config
from open_r1_tpu.evaluation import server as eval_server
from open_r1_tpu.evaluation.preflight import check_export_dir
from open_r1_tpu.evaluation.stack import vllm_tpu_image_tag

TIER0 = Path(__file__).parents[1] / "recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml"


def _settings(**overrides):
    """Resolved settings for the default container server."""
    settings = {
        "serve_command": ["scripts/run_vllm_tpu_container.sh"],
        "server_image": vllm_tpu_image_tag(),
        "model_path": "artifacts/model",
        "served_model_name": "model",
        "host": "127.0.0.1",
        "port": 8000,
        "tensor_parallel_size": 1,
        "max_model_len": None,
        "server_extra_args": [],
    }
    settings.update(overrides)
    return settings


def test_the_default_server_runs_the_derived_local_tpu_image():
    command = eval_server.vllm_serve_command(_settings())

    assert command[:4] == [
        "scripts/run_vllm_tpu_container.sh",
        "--image",
        vllm_tpu_image_tag(),
        "--",
    ]
    assert command[4] == "artifacts/model"


@pytest.mark.parametrize(
    "serve_command",
    [["/opt/vllm-venv/bin/vllm", "serve"], ["docker", "run", "--rm", "img", "serve"]],
    ids=["venv", "docker"],
)
def test_a_custom_serve_command_precedes_the_model_path(serve_command):
    # tpu-inference does not support this project's Python, so vLLM runs
    # wherever it is installed rather than being imported from here.
    command = eval_server.vllm_serve_command(
        _settings(serve_command=serve_command, server_image=None)
    )

    assert command[: len(serve_command)] == serve_command
    assert command[len(serve_command)] == "artifacts/model"


def test_the_server_disables_prefix_caching():
    # A prefix-cache hit changes the prefill's kernel shape and therefore the
    # bf16 logits, so greedy completions would depend on server cache state.
    assert "--no-enable-prefix-caching" in eval_server.vllm_serve_command(_settings())


def test_serve_command_carries_the_recipe_port_and_window():
    command = eval_server.vllm_serve_command(_settings(port=9001, max_model_len=20480))

    assert command[command.index("--port") + 1] == "9001"
    assert command[command.index("--max-model-len") + 1] == "20480"
    assert command[command.index("--served-model-name") + 1] == "model"


# --- integration: needs a live vLLM server -----------------------------------
#
# Start one with scripts/run_eval_tpu.sh, or point OPEN_R1_TPU_EVAL_URL at a
# server that is already up, then run `pytest -m integration`.


@pytest.fixture
def live_settings(tmp_path):
    """Settings pointed at a running server, with the work kept tiny."""
    settings = eval_config.resolve_settings(eval_config.load_eval_config(TIER0))
    base_url = os.environ.get("OPEN_R1_TPU_EVAL_URL")
    if base_url:
        settings["base_url"] = base_url
    settings["max_samples"] = 1
    settings["output_dir"] = str(tmp_path)
    settings["summary_path"] = str(tmp_path / "summary.json")
    settings["wandb"] = {"enabled": False}
    return settings


@pytest.mark.integration
def test_the_served_model_answers_before_a_benchmark_is_committed_to_it(
    live_settings,
):
    eval_server.wait_for_server(live_settings["base_url"], timeout_secs=120)

    with urllib.request.urlopen(
        live_settings["base_url"].rstrip("/") + "/models", timeout=30
    ) as response:
        served = json.loads(response.read())

    # Requests name the model this way, so vLLM must serve it under that name.
    assert live_settings["served_model_name"] in {
        entry["id"] for entry in served["data"]
    }


@pytest.mark.integration
def test_the_configured_model_export_passes_preflight(live_settings):
    assert (
        check_export_dir(live_settings["model_path"], live_settings["turn_end_token"])
        == []
    )
