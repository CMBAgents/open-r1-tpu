# AGENTS.md

## Scope

These instructions apply to the entire repository.

## Project purpose

`open-r1-tpu` implements the supervised fine-tuning stage of a reasoning
distillation pipeline on Google TPUs. It follows the broad SFT-to-GRPO workflow
from Open-R1 while using JAX, Grain, Orbax, Optax, and Google Tunix instead of
CUDA, PyTorch, or TRL.

The default path is:

1. Load conversational reasoning traces from
   `open-r1/Mixture-of-Thoughts`.
2. Apply the Qwen chat template and supervise only the assistant trace.
3. Fine-tune `Qwen/Qwen3-1.7B-Base` with LoRA on one 32 GiB TPU v6e device.
4. Save resumable Tunix/Orbax checkpoints.
5. Merge the LoRA weights into a Hugging Face-style safetensors model for a
   later Tunix GRPO run.

## Repository map

- `src/open_r1_tpu/config.py`: YAML loading, dotted overrides, and validation.
- `src/open_r1_tpu/data.py`: message normalization, chat templating, token
  masking, and Grain datasets.
- `src/open_r1_tpu/sft.py`: model creation, LoRA SFT, optimizer, checkpointing,
  and merged export.
- `src/open_r1_tpu/check_env.py`: target-TPU environment preflight.
- `src/open_r1_tpu/transcripts.py`: periodic free-running sample transcripts.
- `src/open_r1_tpu/evaluate.py`: benchmark evaluation -- recipe validation, the
  LightEval harness run, and reduction of its output to a summary.
- `src/open_r1_tpu/benchmark_generation.py`: controlled, fixed-token throughput
  comparison between the vLLM service and Tunix's direct sampler.
- `src/open_r1_tpu/check_eval_env.py`: inference-environment, export, and
  task-name preflight.
- `recipes/`: versioned model, training, and evaluation configurations.
- `scripts/setup_tpu_vm.sh`: uv-based TPU VM environment provisioning.
- `scripts/copy_gcs_bucket_data.sh`: copy GCS bucket data to local disk.
- `scripts/run_sft_tpu.sh`: standard SFT launcher.
- `scripts/run_eval_tpu.sh`: evaluation launcher; owns the vLLM server's
  lifecycle.
- `scripts/run_vllm_tpu_container.sh`: pinned vLLM TPU container boundary;
  owns device/cache mounts and Docker-level cleanup.
- `scripts/benchmark_generation_tpu.sh`: runs vLLM and Tunix sequentially on
  one TPU and writes their speed comparison.
- `tests/`: unit and integration tests. They run on the TPU VM.

## Architectural invariants

- Keep the training path TPU-native. Do not introduce CUDA, PyTorch, TRL,
  Accelerate, DeepSpeed, or GPU vLLM dependencies into the SFT stage.
- Keep heavyweight JAX/Tunix imports inside runtime functions where practical,
  so that importing a module does not initialize the TPU.
- Preserve assistant-only loss by default. The prompt boundary must come from
  an exact chat-template prefix; do not guess it from string lengths or special
  token IDs.
- Derive valid attention positions from the supervised sequence boundary. Do
  not assume padding and EOS have different token IDs.
- Require complete `<think>...</think>` traces by default, and filter overlength
  examples by default instead of truncating away reasoning or final answers.
  `dataset.overlength_policy: truncate` is opt-in, for corpora whose traces were
  themselves generated under a context cap and are mostly incomplete; it must
  leave the truncated sequence unterminated, since appending a terminator would
  teach the model to stop mid-reasoning.
- Keep integer-label cross-entropy. Tunix's default vocabulary-sized one-hot
  target is unnecessarily expensive at long sequence lengths.
- If LoRA is requested, fail when no model modules match the configured regex.
  Never silently fall back to full-model fine-tuning.
- Preserve denominator-aware `LossOutput`/`WeightedMetric` normalization so
  gradient accumulation weights tokens correctly across microbatches.
- Keep *scalar* W&B logging restricted to stepped `train/*` and `eval/*`
  metrics. Raw global JAX/Orbax events may omit `step`; Metrax maps those
  events to step zero and causes W&B to discard them after training advances.
  Transcript tables are text and cannot travel through that scalar path at all,
  so they are logged straight to the W&B run with the real training step, which
  preserves the ordering the scalar filter exists to protect.
- Keep qualitative sampling optional and non-fatal. Free-running generation adds
  a decode compilation and a KV cache to a validated single-device memory
  profile, so it stays disabled by default, and any sampling failure must
  disable transcripts and let training continue rather than end the run.
- Treat checkpoint, model-cache, dataset, and export paths as potentially
  large. Keep `artifacts/`, `data/`, and `models/` untracked.
- Maintain the export-path safety checks. Merged export must never replace the
  repository, home directory, base-model cache, or checkpoint directory.
- Keep `open_r1_tpu.evaluate` free of JAX, Tunix, and vLLM imports. It drives
  LightEval as a subprocess and the server over a socket, so it stays coupled
  to their command line and wire format rather than to their Python API, both
  of which move faster. Only one process can hold the chip, so evaluation runs
  after training, not beside it.
- Keep vLLM out of the project's dependencies. It is a service this package
  invokes, not a library it imports, and its inference stack does not belong in
  the Python 3.13 LightEval/training environment. Use the immutable official TPU
  image through `server.serve_command` by default; an external environment must
  set `server.image=null` and is reported as reproducibility-unchecked.
- Treat the evaluation environment as a protocol. Keep `.python-version`, the
  exact direct pins in the `eval` extra, `uv.lock`,
  `open_r1_tpu.evaluation_stack`, and the recipe's vLLM image digest in sync.
  Never use a mutable container tag for a reported result.
- Never report a benchmark number from a single seed. Seed variance alone moves
  small reasoning benchmarks by 5-15 points, so results carry a mean and a
  standard deviation, and one seed reports a null spread rather than `0.0`.
- Keep generation-level metrics honest about missing data. Truncation rate and
  mean completion length are `null` when LightEval's detail shards carry no
  token counts; do not substitute a character-length estimate.
- Treat LightEval's CLI and detail-column names as unstable. Task strings and
  extra flags belong in the recipe, and detail fields are probed with a failure
  that names the keys that were actually present.

## Tunix compatibility

Tunix is pinned to an exact Git commit in `pyproject.toml`. The pin is a Git
commit rather than a PyPI release for two reasons:

- The code relies on APIs that exist only on `main`. Concretely,
  `WeightedMetric` — which the custom loss in `src/open_r1_tpu/sft.py` returns
  so that loss sums and denominators aggregate correctly across gradient
  accumulation — is absent from the newest release (`v0.1.7`, checked
  2026-08-18), so installing any released version breaks training at the first
  step. Tunix cuts releases roughly quarterly while `main` moves daily.
- Given a Git dependency, an exact hash rather than a branch ref keeps the
  installed code byte-stable: `@main` re-resolves on every install, so two VMs
  set up days apart would silently run different Tunix. Training runs are
  recorded with verbatim launch commands, and that reproducibility is only
  meaningful if the environment is fixed.

Do not move the pin casually; nothing currently upstream earns it. The two
upstream changes that would are a native full-model (non-LoRA) safetensors
saver, which would replace `src/open_r1_tpu/safetensors_export.py`, and the
sampler passing `segment_ids` into Qwen splash attention, which would fix
padded splash inference. (The pinned model itself already accepts
`segment_ids`; training-side packing uses that directly through the custom
`gen_model_input`/loss in `src/open_r1_tpu/sft.py` — the gap is only that the
sampler never supplies them.) Any pin update requires a fresh review of:

- `PeftTrainer`, `TrainingConfig`, and `with_loss_fn`;
- `TrainingInput`, `LossOutput`, and `WeightedMetric`;
- model and tokenizer creation helpers;
- Qwen3 internal LoRA module paths;
- Qwen3 merged-LoRA safetensors export;
- the Qwen3 loader key/transform mapping in `tunix/models/qwen3/params.py`,
  which `src/open_r1_tpu/safetensors_export.py` inverts; and
- checkpoint option construction.

Do not claim that training is TPU-compatible merely because the unit suite
passes. Runtime compatibility requires the TPU-side preflight and a compiled
smoke run.

## Environment and Hugging Face

- Use standard CPython 3.13 on the TPU VM; `.python-version` pins the tested
  patch release, and the free-threaded `3.13t` build is unsupported.
- Install with `python -m pip install -e '.[test]'` on the TPU VM. The Tunix
  dependency installs `jax[tpu]`, so avoid treating an unrelated local virtual
  environment as authoritative.
- Set `HF_TOKEN` in the environment. Never print, commit, log, or embed token
  values in commands or configuration.
- When model and dataset artifacts are already staged in GCS, copy them to
  ignored local `models/` and `data/` directories. Use `model_source=local` and
  the Parquet builder with `dataset.data_files`; do not redownload them from the
  Hub.
- Use the current `hf` CLI, not the deprecated `huggingface-cli`. A safe
  authentication check is `hf auth whoami`.
- Do not upload models, datasets, checkpoints, or traces to the Hub unless the
  user explicitly requests the upload and identifies the destination.

## Development commands

Host-independent checks:

```bash
python -m pytest
python -m compileall -q src tests
for script in scripts/*.sh; do bash -n "$script"; done
git diff --check
```

Lint, format, and type checks run through pre-commit. Install the dev extra and
the hook once per clone:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

The hooks then run on every commit, and across the whole tree on demand:

```bash
pre-commit run --all-files
```

The same tools can be driven directly:

```bash
ruff check .
ruff format .
pyright
```

Ruff and pyright settings live in `pyproject.toml`. Pyright runs in `standard`
mode and resolves imports from `.venv`, so keep that environment installed with
the `dev`, `test`, and `eval` extras. On the TPU VM every import resolves and an
unresolved-import warning is a real finding.

Target-TPU preflight:

```bash
python -m open_r1_tpu.check_env
```

Short TPU smoke run:

```bash
./scripts/run_sft_tpu.sh \
  dataset.max_examples=128 \
  training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2
```

Full run:

```bash
./scripts/run_sft_tpu.sh
```

Evaluation preflight and smoke tier, with the `eval` extra installed:

```bash
./scripts/setup_tpu_vm.sh --with-eval
python -m open_r1_tpu.check_eval_env \
  --config recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml
RECIPE=recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

The first TPU step includes JAX/XLA compilation and can be much slower than
subsequent steps.

## Testing expectations

- Add or update unit tests for changes to configuration parsing, message
  validation, tag filtering, token boundaries, padding, or loss masking, or to
  evaluation recipe validation, command construction, or metric reduction.
- Test against the real thing. Use the configured Qwen tokenizer, real Parquet
  shards, and a real served model rather than fakes; the suite runs on the TPU
  VM, where the whole stack is installed. Reach for a stub only where the real
  dependency cannot be reached at all, and say so at the test.
- Mark tests that need a live vLLM server `@pytest.mark.integration`. They are
  deselected by default and run with `pytest -m integration` once a server is
  up.
- Run the smallest relevant tests during development, then the complete unit
  suite before handoff.
- When changing model topology, LoRA paths, sequence length, sharding, remat,
  flash attention, optimizer behavior, or checkpointing, also run the TPU
  preflight and smoke job.
- Report validation precisely. Distinguish source inspection, the unit suite,
  the integration suite, TPU preflight, JAX compilation, completed optimizer
  steps, checkpoint writes, and merged export; they are not interchangeable
  evidence.

## Configuration guidance

- Keep reusable defaults in YAML recipes and expose experiment-specific values
  through dotted command-line overrides.
- The product of `model.mesh.shape` must equal the number of visible JAX
  devices, and `axis_names` must have the same rank as `shape`.
- Keep batch and sharding choices compatible with the target topology.
- Increase `dataset.max_length` only after observing HBM use on the target TPU.
  The default 1024 is the validated single-device baseline; also measure how
  many complete examples remain after overlength filtering.
- Keep a finite `training.max_steps` for predictable resume and checkpoint
  behavior. Ensure `num_train_epochs` supplies enough examples after filtering.
- When changing model families, disable merged export unless the corresponding
  Tunix params module implements
  `save_lora_merged_model_as_safetensors`.
- Orbax checkpoints can target GCS, but merged safetensors export must target a
  local directory and be synced to GCS only after export completes.

## Code and change discipline

- Target Python 3.13 and prefer typed, small functions with explicit failure
  messages for conditions that would waste TPU time.
- Preserve existing user changes and avoid unrelated refactors.
- Keep committed defaults neutral and deployment-independent. W&B entity and
  project names, bucket names, hostnames, and paths belong in the environment
  or in dotted command-line overrides, not in tracked files.
- Do not commit generated datasets, downloaded model weights, checkpoints,
  logs, profiler output, secrets, or merged artifacts.
- Update `README.md`, the recipe, tests, and preflight together when a behavior
  or operator-facing command changes.
- Do not start a full training run, download large artifacts, publish outputs,
  or delete checkpoints without explicit user authorization.
