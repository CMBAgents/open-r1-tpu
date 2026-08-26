"""Repair tpu-inference's eager ``getattr`` fallback onto ``hf_config.text_config``.

``tpu_inference`` reads hidden sizes and similar shape parameters with

    getattr(model_config.hf_config, "hidden_size",
            model_config.hf_config.text_config.hidden_size)

Python evaluates a ``getattr`` default eagerly, before deciding whether the
attribute is missing, so ``.text_config`` is always dereferenced. Only
multimodal Hugging Face configs carry ``text_config``; a flat text config such
as ``Qwen2Config`` raises ``AttributeError`` and takes the engine core down
during startup, even though it holds the wanted attribute directly.

``PretrainedConfig.get_text_config()`` expresses the intended lookup in one
step: it returns the nested text config when there is one and the config itself
otherwise. This script rewrites the broken idiom to use it.

Run at image build time, against the installed package; stdlib only. It fails
the build when the idiom has disappeared or moved, so an upstream change is
reviewed rather than silently unpatched.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

# getattr(<config>, "<attribute>", <config>.text_config.<attribute>), with the
# backreferences requiring the same config expression and attribute name in both
# positions, and \s* absorbing the line break upstream wraps the call on.
EAGER_FALLBACK = re.compile(
    r"getattr\(\s*"
    r"(?P<config>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*,\s*"
    r"(?P<quote>['\"])(?P<attribute>[A-Za-z_]\w*)(?P=quote)\s*,\s*"
    r"(?P=config)\.text_config\.(?P=attribute)\s*,?\s*\)"
)
REPLACEMENT = r"\g<config>.get_text_config().\g<attribute>"


def package_root() -> Path:
    """Locate the installed ``tpu_inference`` package without importing it."""
    spec = importlib.util.find_spec("tpu_inference")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("tpu_inference is not installed in this environment")
    return Path(next(iter(spec.submodule_search_locations)))


def patch_source(source: str) -> tuple[str, int]:
    return EAGER_FALLBACK.subn(REPLACEMENT, source)


def main() -> int:
    root = package_root()
    patched_files = 0
    patched_sites = 0

    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ".text_config." not in source:
            continue
        updated, count = patch_source(source)
        if not count:
            continue
        # Refuse to ship a file the rewrite has broken.
        compile(updated, str(path), "exec")
        path.write_text(updated, encoding="utf-8")
        # Drop the byte-code the installer precompiled from the broken source.
        for cached in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
            cached.unlink()
        patched_files += 1
        patched_sites += count
        print(f"patched {count} site(s) in {path.relative_to(root)}")

    if not patched_sites:
        raise SystemExit(
            "No eager text_config fallback found in tpu_inference. The upstream "
            "bug this patch works around has moved or been fixed; re-review the "
            "patch against the pinned tpu-inference before rebuilding."
        )

    print(
        f"lazy_text_config_fallback: {patched_sites} site(s) in {patched_files} file(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
