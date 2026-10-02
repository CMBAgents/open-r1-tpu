"""The tutorial in examples/: its recipes, notebooks and helper scripts."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.grpo.config import validate_grpo_config
from open_r1_tpu.sft.config import validate_sft_config

REPO = Path(__file__).parents[1]
EXAMPLES = REPO / "examples"
SFT_RECIPE = EXAMPLES / "recipes/sft-gsm8k.yaml"
GRPO_RECIPE = EXAMPLES / "recipes/grpo-gsm8k.yaml"
NOTEBOOKS = sorted(EXAMPLES.glob("*.ipynb"))


def _load_module(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sft() -> dict[str, Any]:
    return load_config(SFT_RECIPE, [], validator=validate_sft_config)


def _grpo() -> dict[str, Any]:
    return load_config(GRPO_RECIPE, [], validator=validate_grpo_config)


@pytest.mark.parametrize("chips", [1, 2, 4, 8])
def test_recipes_validate_with_the_notebooks_mesh_override(chips):
    override = [f"model.mesh.shape=[{chips},1]"]
    sft = load_config(SFT_RECIPE, override, validator=validate_sft_config)
    grpo = load_config(GRPO_RECIPE, override, validator=validate_grpo_config)
    # Each chip on the fsdp axis takes an equal share of the batch.
    assert sft["dataset"]["batch_size"] % chips == 0
    assert grpo["grpo"]["num_generations"] % chips == 0


def test_grpo_starts_from_the_sft_export_with_its_data_and_prompt():
    sft, grpo = _sft(), _grpo()
    assert grpo["model"]["model_path"] == sft["export"]["output_dir"]
    assert grpo["tokenizer"]["tokenizer_path"] == sft["export"]["output_dir"]
    assert grpo["model"]["model_name"] == sft["model"]["model_name"]
    assert grpo["model"]["rope_theta"] == sft["model"]["rope_theta"]
    assert grpo["dataset"]["data_files"] == sft["dataset"]["data_files"]
    prompt = sft["dataset"]["system_prompt_file"]
    assert grpo["dataset"]["system_prompt_file"] == prompt
    assert (REPO / prompt).is_file()


def test_grpo_recipe_keeps_the_settings_qwen2_rollouts_need():
    grpo = _grpo()
    assert grpo["model"]["dtype"] == "float32"
    assert grpo["model"]["use_flash_attention"] is False
    rollout = grpo["rollout"]
    # The pinned sampler decodes greedily without top_p, and a group of
    # identical answers has no advantage to learn from.
    assert rollout["top_p"] is not None
    assert rollout["temperature"] > 0
    assert rollout["max_prompt_length"] >= grpo["dataset"]["max_prompt_length"]
    assert rollout["kv_cache_size"] >= (
        rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    )


def test_tutorial_outputs_stay_under_artifacts():
    for config in (_sft(), _grpo()):
        training = config["training"]
        for path in (
            training["checkpoint_dir"],
            training["metrics_log_dir"],
            config["export"]["output_dir"],
        ):
            assert path.startswith("artifacts/")
        assert config["training"]["wandb"]["enabled"] is False


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebooks_are_committed_without_outputs(notebook):
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    for cell in code_cells:
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


def test_notebooks_launch_the_tutorial_recipes():
    sources = {
        notebook.name: "".join(
            "".join(cell["source"])
            for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"]
        )
        for notebook in NOTEBOOKS
    }
    assert str(SFT_RECIPE.relative_to(REPO)) in sources["01_sft.ipynb"]
    assert str(GRPO_RECIPE.relative_to(REPO)) in sources["02_grpo.ipynb"]


answer_questions = _load_module("answer_questions")


@pytest.mark.parametrize(
    ("reply", "correct", "formatted"),
    [
        ("<think>\n6 * 7 = 42\n</think>\nThe answer is \\boxed{42}.", True, True),
        ("<think>\n6 * 7 = 41\n</think>\nThe answer is \\boxed{41}.", False, True),
        # A base model's free-form reply: its last number is its answer.
        ("Six sevens make 42.", True, False),
        ("<think>\n6 * 7 =", False, False),
    ],
)
def test_score_gives_the_verdicts_and_every_reward(reply, correct, formatted):
    scored = answer_questions.score(reply, "42")
    assert scored["correct"] is correct
    assert scored["formatted"] is formatted
    assert set(scored["rewards"]) == {
        "format_reward",
        "correctness_reward",
        "answer_correctness_reward",
    }


def test_batches_have_one_shape_and_count_their_real_items():
    batches = answer_questions.batches(["a", "b", "c", "d", "e"], 2)
    assert batches == [(["a", "b"], 2), (["c", "d"], 2), (["e", "e"], 1)]


def test_strip_turn_end_removes_the_markers_the_sampler_leaves():
    assert answer_questions.strip_turn_end("\\boxed{4}<|im_end|>") == "\\boxed{4}"
    assert answer_questions.strip_turn_end(" 4 <|endoftext|>") == "4"


def test_read_scalars_reads_back_what_tensorboard_x_wrote(tmp_path):
    pytest.importorskip("pandas")
    tensorboard_x = pytest.importorskip("tensorboardX")
    notebook_utils = _load_module("notebook_utils")

    writer = tensorboard_x.SummaryWriter(logdir=str(tmp_path))
    for step in (1, 2):
        writer.add_scalar("train/loss", 1.0 / step, step)
    writer.add_scalar("train/loss", 0.25, 2)  # logged twice: the last value wins
    writer.close()
    nested = tensorboard_x.SummaryWriter(logdir=str(tmp_path / "actor"))
    nested.add_scalar("rewards/train/sum", 3.0, 1)
    nested.close()

    scalars = notebook_utils.read_scalars(tmp_path)
    rows = {(row.tag, row.step): row.value for row in scalars.itertuples(index=False)}
    assert rows == {
        ("rewards/train/sum", 1): 3.0,
        ("train/loss", 1): 1.0,
        ("train/loss", 2): 0.25,
    }
