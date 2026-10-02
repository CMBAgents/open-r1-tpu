# Development

[AGENTS.md](https://github.com/CMBAgents/open-r1-tpu/blob/main/AGENTS.md)
holds the rules for changing the code: the architectural invariants, the
Tunix pin, and what each kind of change must be checked with.

## Set up

```bash
uv sync --frozen --extra test --extra eval --extra dev
pre-commit install
```

`uv sync` also installs console scripts for the entry points:

| Command | Module |
| --- | --- |
| `open-r1-tpu-sft` | `open_r1_tpu.sft.run` |
| `open-r1-tpu-sft-preflight` | `open_r1_tpu.sft.preflight` |
| `open-r1-tpu-grpo` | `open_r1_tpu.grpo.run` |
| `open-r1-tpu-eval` | `open_r1_tpu.evaluation.run` |
| `open-r1-tpu-eval-preflight` | `open_r1_tpu.evaluation.preflight` |

## Checks

```bash
python -m pytest                       # unit suite, no TPU needed
pre-commit run --all-files             # ruff, ruff format, pyright
for script in scripts/*.sh scripts/lib/*.sh; do bash -n "$script"; done
```

The unit tests need no TPU; tests that need JAX, Tunix or the `eval` extra
skip without them. Two groups are deselected by default:

| Marker | Needs | Run with |
| --- | --- | --- |
| `integration` | A live vLLM server | `python -m pytest -m integration` |
| `network` | The Hugging Face Hub | `python -m pytest -m network` |

Ruff and pyright run as pre-commit hooks, and CI
([`.github/workflows/ci.yml`](https://github.com/CMBAgents/open-r1-tpu/blob/main/.github/workflows/ci.yml))
runs the hooks and the tests on every push and pull request.

!!! note "A passing unit suite is not a working TPU run"

    The TPU checks are the SFT and evaluation preflights and the four-step
    [smoke run](sft.md#check-smoke-test-train). Run them when changing model
    topology, LoRA paths, sequence length, sharding, remat, flash attention,
    the optimizer or checkpointing.

## Pins

Every dependency is pinned in `uv.lock`, and the evaluation stack is pinned
exactly (`src/open_r1_tpu/evaluation/stack.py`): change the pins, the lock and
`docker/vllm-tpu` together. Tunix is pinned to an exact commit, because the
code needs APIs only on its main branch.

## This site

The pages are the Markdown files in `docs/`, built with
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) from
`mkdocs.yml`. Its tooling is pinned in `docs/requirements.txt`, apart from
the project's lock. Preview it with live reload:

```bash
uvx --with-requirements docs/requirements.txt mkdocs serve
```

CI ([`.github/workflows/docs.yml`](https://github.com/CMBAgents/open-r1-tpu/blob/main/.github/workflows/docs.yml))
builds it with `mkdocs build --strict` on every push and pull request, so a
broken link or anchor fails, and publishes `main` to GitHub Pages. Link to
files outside `docs/` by their full GitHub address, so the link works both
on GitHub and here.
