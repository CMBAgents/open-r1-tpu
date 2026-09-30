"""scripts/export_checkpoint.py, without a TPU: Tunix, model loading, restore
and export are faked, so these check argument handling, recipe reading and
what reaches each stage."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

REPO = Path(__file__).parents[1]
DISTILL_RECIPE = REPO / "recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml"

SPEC = importlib.util.spec_from_file_location(
    "export_checkpoint", REPO / "scripts/export_checkpoint.py"
)
assert SPEC is not None and SPEC.loader is not None
export_script = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export_script)


def _lora_recipe(tmp_path: Path) -> Path:
    recipe = tmp_path / "lora.yaml"
    recipe.write_text(
        f"extends: {DISTILL_RECIPE}\n"
        "model:\n"
        "  lora_config: {module_path: '.*q_proj', rank: 64, alpha: 64.0}\n"
        "training:\n"
        "  checkpoint_dir: artifacts/lora-run/checkpoints\n"
    )
    return recipe


@pytest.mark.parametrize("missing", ["--recipe", "--step", "--output"])
def test_recipe_step_and_output_are_required(missing, capsys):
    argv = {"--recipe": "r.yaml", "--step": "200", "--output": "out"}
    del argv[missing]

    with pytest.raises(SystemExit):
        export_script.parse_args([item for pair in argv.items() for item in pair])
    assert missing in capsys.readouterr().err


def test_step_must_be_an_integer():
    with pytest.raises(SystemExit):
        export_script.parse_args(
            ["--recipe", "r.yaml", "--step", "latest", "--output", "out"]
        )


def test_a_full_finetune_recipe_restores_every_parameter_into_its_export():
    args = export_script.parse_args(
        ["--recipe", str(DISTILL_RECIPE), "--step", "200", "--output", "out/merged"]
    )

    config, lora_config, checkpoint_dir = export_script.read_recipe(args)

    assert lora_config is None
    assert checkpoint_dir.endswith("OpenR1-Distill-Qwen2.5-Math-1.5B/checkpoints")
    assert config["export"]["output_dir"] == "out/merged"
    assert config["export"]["enabled"] is True
    assert config["export"]["overwrite"] is True


def test_a_lora_recipe_supplies_its_adapter_geometry(tmp_path):
    args = export_script.parse_args(
        ["--recipe", str(_lora_recipe(tmp_path)), "--step", "1", "--output", "out"]
    )

    _config, lora_config, checkpoint_dir = export_script.read_recipe(args)

    assert lora_config is not None
    assert lora_config["rank"] == 64
    assert lora_config["alpha"] == 64.0
    assert checkpoint_dir == "artifacts/lora-run/checkpoints"


def test_checkpoint_dir_overrides_the_recipe():
    args = export_script.parse_args(
        [
            "--recipe",
            str(DISTILL_RECIPE),
            "--step",
            "200",
            "--output",
            "out",
            "--checkpoint-dir",
            "gs://bucket/run/checkpoints",
        ]
    )

    assert export_script.read_recipe(args)[2] == "gs://bucket/run/checkpoints"


@pytest.mark.parametrize("lora", [False, True])
def test_main_restores_what_the_recipe_trained_and_exports_it(
    tmp_path, monkeypatch, lora
):
    recipe = _lora_recipe(tmp_path) if lora else DISTILL_RECIPE
    model = object()
    captured: dict[str, Any] = {}

    fake_model_utils: Any = ModuleType("tunix.cli.utils.model")
    fake_model_utils.create_tokenizer = lambda config, path: SimpleNamespace(
        config=config, path=path
    )
    for name, module in {
        "tunix": ModuleType("tunix"),
        "tunix.cli": ModuleType("tunix.cli"),
        "tunix.cli.utils": ModuleType("tunix.cli.utils"),
        "tunix.cli.utils.model": fake_model_utils,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    def create_model(config, mesh):
        captured["mesh"] = mesh
        return model, "/models/base"

    def restore_checkpoint(restored_model, checkpoint_dir, step, *, lora_only):
        captured["restore"] = (restored_model, checkpoint_dir, step, lora_only)
        return step

    def export_model(*, config, model, tokenizer, local_model_path):
        captured["export"] = (config["export"], model, tokenizer, local_model_path)

    def create_mesh(config):
        mesh = config["model"]["mesh"]
        return ("mesh", tuple(mesh["shape"]), tuple(mesh["axis_names"]))

    monkeypatch.setattr(export_script, "create_mesh", create_mesh)
    monkeypatch.setattr(export_script, "create_model", create_model)
    monkeypatch.setattr(export_script, "restore_checkpoint", restore_checkpoint)
    monkeypatch.setattr(export_script, "export_model", export_model)

    export_script.main(
        ["--recipe", str(recipe), "--step", "200", "--output", "out/merged"]
    )

    assert captured["mesh"] == ("mesh", (2, 2), ("fsdp", "tp"))
    assert captured["restore"][0] is model
    assert captured["restore"][2:] == (200, lora)
    export, exported_model, tokenizer, local_model_path = captured["export"]
    assert export["output_dir"] == "out/merged"
    assert exported_model is model
    assert local_model_path == "/models/base"
    assert tokenizer.config["tokenizer_path"] == "/models/base"
    assert tokenizer.config["chat_template"] is None
