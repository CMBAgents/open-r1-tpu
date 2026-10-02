# AGENTS.md

These instructions apply to the whole repository. [README.md](README.md) says
what the project does and how to run it; [docs/](docs/) holds the operating
guides. This file holds the rules for changing the code.

## Repository map

- `src/open_r1_tpu/core/`: recipe loading and validation, the shared command
  line (`cli.py`), logging, installed-package lookups.
- `src/open_r1_tpu/model/`: model and tokenizer creation, LoRA, optimizer,
  metrics logging, checkpoint restore and merged export, shared by SFT and
  GRPO.
- `src/open_r1_tpu/sft/`: recipe schema (`config.py`), data preparation and
  packing, transcripts, training (`run.py`) and the TPU preflight.
- `src/open_r1_tpu/grpo/`: recipe schema, prompt loading, rewards and training.
- `src/open_r1_tpu/evaluation/`: recipe schema and settings (`config.py`), the
  vLLM command and readiness wait (`server.py`), generation (`generate.py`),
  scoring (`scoring.py`), records and the summary (`summary.py`), cons@n,
  optional Langfuse tracing (`traced.py`), the entry point (`run.py`),
  LightEval task lookup (`tasks.py`), the stack pins and the preflight.
- `recipes/<base model>/{sft,grpo,eval}/`: one YAML per dataset (training) or
  tier (evaluation).
- `scripts/`: shell launchers (`setup_tpu_vm.sh`, `run_eval_tpu.sh`,
  `run_vllm_tpu_container.sh`, with shared helpers in `scripts/lib/`) and
  standalone tools (chat, completion, checkpoint export, RoPE editing,
  staging a model for evaluation, Langfuse setup).
- `docker/vllm-tpu/`: the pinned vLLM TPU image. `docker/langfuse/`: the
  optional self-hosted Langfuse stack.
- `configs/`: the template for the optional Langfuse tracing config.
- `docs/` and `mkdocs.yml`: the documentation site, built with Material for
  MkDocs; its tooling is pinned in `docs/requirements.txt`, apart from
  `uv.lock`. Pages link to files outside `docs/` by full GitHub URL.
- `examples/`: the tutorial notebooks, with their own recipes in
  `examples/recipes/` and helper scripts. Notebooks are committed without
  outputs; they run training as separate commands and never import JAX.
- `tests/`: unit tests, plus `integration` (live vLLM server) and `network`
  (Hugging Face Hub) tests that are deselected by default.

## Architectural invariants

Training:

- Keep the training path TPU-native. Do not add CUDA, PyTorch, TRL,
  Accelerate, DeepSpeed or GPU vLLM dependencies.
- Keep JAX and Tunix imports inside runtime functions where practical, so that
  importing a module does not initialise the TPU.
- Preserve assistant-only loss by default. The prompt boundary must come from
  an exact chat-template prefix, never from string lengths or special token
  IDs.
- Derive valid attention positions from the supervised sequence boundary. Do
  not assume padding and EOS have different token IDs.
- Require complete `<think>...</think>` traces by default, and drop overlength
  examples rather than truncate them. `dataset.overlength_policy: truncate` is
  opt-in, for corpora whose traces were themselves cut off; it must leave the
  truncated sequence unterminated, since a terminator would teach the model to
  stop mid-reasoning.
- Keep integer-label cross-entropy. Tunix's default one-hot target is
  vocabulary-sized and expensive at long sequence lengths.
- If LoRA is requested, fail when no module matches the configured regex.
  Never fall back silently to full fine-tuning.
- Preserve denominator-aware `LossOutput`/`WeightedMetric` normalisation, so
  gradient accumulation weights tokens correctly across microbatches.
- Log only stepped `train/*` and `eval/*` scalars through the metrics path.
  JAX and Orbax events without a step land at step zero and W&B discards them
  once training has advanced. Transcript tables go straight to the W&B run
  with the real training step.
- Keep transcript sampling optional and non-fatal. It adds a decode
  compilation and a KV cache to a validated memory profile, so it is off by
  default, and a sampling failure disables transcripts rather than ending the
  run.
- Keep the export-path safety checks. A merged export must never replace the
  repository, the home directory, the base-model cache or the checkpoint
  directory.

Evaluation:

- Keep `open_r1_tpu.evaluation` free of JAX, Tunix and vLLM imports. vLLM is a
  service reached over HTTP, not a library: it stays out of the project's
  dependencies and runs from the image built from `docker/vllm-tpu`. An
  external server sets `server.image=null` and is reported as
  reproducibility-unchecked.
- LightEval is used only as a library: its task registry, `Doc` and
  `ModelResponse` types and metrics. Never its runner, models or litellm
  client. `tasks.py` and `scoring.py` name the internals they rely on.
- Keep Langfuse optional. Without `--tracing-config` an evaluation needs
  nothing but the vLLM server, and the local and Langfuse paths must write
  identical JSONL records.
- Score on the main thread: LightEval's maths metrics time out with
  `signal.alarm`, which works only there.
- Treat the evaluation environment as part of the protocol. Keep
  `.python-version`, the exact pins in the `eval` extra, `uv.lock`,
  `open_r1_tpu.evaluation.stack` and `docker/vllm-tpu`'s base digest and lock
  in step. Never report a result from a mutable container tag.
- Never report a benchmark number from a single seed. Seed variance alone
  moves small reasoning benchmarks by 5-15 points, so results carry a mean and
  a standard deviation, and one seed reports a null spread, not `0.0`.
- Keep generation statistics honest about missing data. Truncation rate and
  completion length are `null` when the server returned no finish reason or
  token count; never estimate them from characters.

## Tunix pin

Tunix is pinned to an exact Git commit in `pyproject.toml`, not a PyPI
release:

- The code needs APIs that exist only on `main`. `WeightedMetric`, which the
  SFT loss returns so that loss sums and denominators aggregate across
  gradient accumulation, is absent from the newest release (`v0.1.7`, checked
  2026-08-18).
- An exact hash keeps the installed code identical across VMs; a branch ref
  would re-resolve on every install.

Do not move the pin without a reason upstream. Two changes would earn it: a
native full-model safetensors saver, which would replace
`model/export.py`'s own mapping, and the sampler passing `segment_ids` to
Qwen splash attention, which would fix padded splash inference. Any pin update
needs a fresh review of:

- `PeftTrainer`, `TrainingConfig` and `with_loss_fn`;
- `TrainingInput`, `LossOutput` and `WeightedMetric`;
- model and tokenizer creation helpers;
- the Qwen2 LoRA module paths;
- the Qwen2 and Qwen3 loader key mappings in `tunix/models/*/params.py`,
  which `model/export.py` inverts, and the merged-LoRA export
  `model/export.py` supplies for Qwen2;
- checkpoint option construction;
- the GRPO dtype and attention constraints in the README.

A passing unit suite does not show that training works on TPU; that needs the
preflight and a compiled smoke run.

## Environment

- Use standard CPython 3.13 at the patch release in `.python-version`; the
  free-threaded `3.13t` build is unsupported.
- Install with `./scripts/setup_tpu_vm.sh`, or `uv sync --frozen` with the
  extras you need (`test`, `eval`, `dev`). The Tunix dependency pulls in
  `jax[tpu]`, so a virtual environment off the TPU VM is not authoritative.
- Read `HF_TOKEN` from the environment. Never print, log, commit or embed a
  token.
- When models and datasets are already in a GCS bucket, copy them to the
  ignored `models/` and `data/` directories and use `model_source: local`
  rather than downloading them again.
- Use the `hf` CLI, not the deprecated `huggingface-cli`.
- Do not upload models, datasets, checkpoints or traces anywhere unless the
  user asks and names the destination.

## Checks

```bash
python -m pytest                       # unit suite, no TPU needed
pre-commit run --all-files             # ruff, ruff format, pyright
for script in scripts/*.sh scripts/lib/*.sh; do bash -n "$script"; done
uvx --with-requirements docs/requirements.txt mkdocs build --strict   # docs site
```

Pyright resolves imports from `.venv`, so install the `dev`, `test` and `eval`
extras there; on the TPU VM an unresolved-import warning is a real finding.
Run `python -m pytest -m integration` with a vLLM server up, and
`python -m pytest -m network` with Hub access.

The TPU checks are the SFT and evaluation preflights and the four-step smoke
run in the README's quick start. The first step includes XLA compilation and
is much slower than the rest.

## Tests

- Add or update tests when changing recipe validation, message handling, tag
  filtering, token boundaries, padding, loss masking, command construction,
  scoring or metric reduction.
- Test against the real thing: the configured tokenizer, real Parquet shards,
  a real served model. Stub only what cannot be reached at all, and say so at
  the test.
- Mark tests that need a live vLLM server `integration` and tests that need
  the Hub `network`. Skip tests that need the `eval` extra with
  `pytest.importorskip`, so the training-only environment still collects.
- When changing model topology, LoRA paths, sequence length, sharding, remat,
  flash attention, the optimizer or checkpointing, also run the preflight and
  the smoke run.
- Report validation precisely. Source inspection, the unit suite, the
  integration suite, the preflight, compilation, completed optimizer steps,
  checkpoint writes and merged export are different evidence.

## Recipes

- Lay recipes out as `recipes/<base model>/<stage>/<dataset>.yaml`, where the
  directory is the Hugging Face name of the model the run starts from.
  Evaluation tiers are `recipes/<model>/eval/tier<N>_<name>.yaml` and extend
  `base.yaml`. Outputs go under `artifacts/<output model name>/`.
- Recipes reject unknown keys. Add a new key to its schema, with a test: the
  stage's own (`sft/config.py`, `grpo/config.py`, `evaluation/config.py`) or
  a shared one (`model/optimizer.py`, `model/metrics.py` for metrics and
  `training.wandb`, `model/export.py`). `model`, `tokenizer` and
  `training.checkpointing_options` pass through to Tunix unchecked.
- The product of `model.mesh.shape` must equal the number of visible JAX
  devices, and `axis_names` must have the same rank.
- Raise `dataset.max_length` only after measuring HBM on the target TPU, and
  check how many examples survive overlength filtering.
- Keep a finite `training.max_steps`, and check `num_train_epochs` supplies
  enough examples after filtering.
- Orbax checkpoints may target GCS; merged export must target a local
  directory and be copied to GCS afterwards.

## Change discipline

- Target Python 3.13. Prefer small typed functions, and fail early with a
  clear message on anything that would waste TPU time.
- Preserve existing user changes and avoid unrelated refactors.
- Keep committed defaults neutral. W&B entities and projects, bucket names,
  hostnames and personal paths belong in the environment or in dotted
  overrides, never in tracked files, including examples and comments.
- Do not commit datasets, model weights, checkpoints, logs, profiler output,
  secrets or merged artifacts. `artifacts/`, `data/` and `models/` are
  ignored.
- When a behaviour or command changes, update the README or the relevant
  `docs/` guide, the recipe, the tests and the preflight together.
- Do not start a full training run, download large artifacts, publish
  outputs or delete checkpoints without the user's explicit permission.
