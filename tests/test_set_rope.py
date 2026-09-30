import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "set_rope.py"
SPEC = importlib.util.spec_from_file_location("set_rope", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
set_rope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(set_rope)


def test_set_rope_edits_only_theta_and_context(tmp_path):
    original = {
        "architectures": ["Qwen2ForCausalLM"],
        "rope_theta": 10000.0,
        "max_position_embeddings": 4096,
        "tie_word_embeddings": True,
    }
    (tmp_path / "config.json").write_text(json.dumps(original))

    set_rope.set_rope(tmp_path, rope_theta=300000, max_position_embeddings=32768)

    written = json.loads((tmp_path / "config.json").read_text())
    assert written == original | {
        "rope_theta": 300000,
        "max_position_embeddings": 32768,
    }
