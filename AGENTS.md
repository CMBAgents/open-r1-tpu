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
3. Fine-tune `Qwen/Qwen3-1.7B-Base` with LoRA on a v5e-8 TPU.
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
- `recipes/`: versioned model and training configurations.
- `scripts/run_sft_tpu.sh`: standard SFT launcher.
- `tests/`: host-independent unit tests.

## Architectural invariants

- Keep the training path TPU-native. Do not introduce CUDA, PyTorch, TRL,
  Accelerate, DeepSpeed, or GPU vLLM dependencies into the SFT stage.
- Keep heavyweight JAX/Tunix imports inside runtime functions where practical.
  Config and preprocessing tests must remain runnable without a TPU stack.
- Preserve assistant-only loss by default. The prompt boundary must come from
  an exact chat-template prefix; do not guess it from string lengths or special
  token IDs.
- Derive valid attention positions from the supervised sequence boundary. Do
  not assume padding and EOS have different token IDs.
- Require complete `<think>...</think>` traces by default. Filter overlength
  examples instead of truncating away reasoning or final answers.
- Keep integer-label cross-entropy. Tunix's default vocabulary-sized one-hot
  target is unnecessarily expensive at long sequence lengths.
- If LoRA is requested, fail when no model modules match the configured regex.
  Never silently fall back to full-model fine-tuning.
- Preserve denominator-aware `LossOutput`/`WeightedMetric` normalization so
  gradient accumulation weights tokens correctly across microbatches.
- Treat checkpoint, model-cache, dataset, and export paths as potentially
  large. Keep `artifacts/`, `data/`, and `models/` untracked.
- Maintain the export-path safety checks. Merged export must never replace the
  repository, home directory, base-model cache, or checkpoint directory.

## Tunix compatibility

Tunix is pinned to an exact Git commit in `pyproject.toml`. Code may rely on the
APIs at that commit, but any pin update requires a fresh review of:

- `PeftTrainer`, `TrainingConfig`, and `with_loss_fn`;
- `TrainingInput`, `LossOutput`, and `WeightedMetric`;
- model and tokenizer creation helpers;
- Qwen3 internal LoRA module paths;
- Qwen3 merged-LoRA safetensors export; and
- checkpoint option construction.

Do not claim that training is TPU-compatible merely because unit tests pass on
a Mac or CPU host. Runtime compatibility requires the TPU-side preflight and a
compiled smoke run.

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
bash -n scripts/run_sft_tpu.sh
git diff --check
```

Target-TPU preflight:

```bash
python -m open_r1_tpu.check_env
```

Short TPU smoke run:

```bash
./scripts/run_sft_tpu.sh \
  dataset.max_examples=128 \
  dataset.max_length=2048 \
  training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2
```

Full run:

```bash
./scripts/run_sft_tpu.sh
```

The first TPU step includes JAX/XLA compilation and can be much slower than
subsequent steps.

## Testing expectations

- Add or update unit tests for changes to configuration parsing, message
  validation, tag filtering, token boundaries, padding, or loss masking.
- Use fake tokenizers for deterministic unit tests. Validate against the real
  configured Qwen tokenizer during TPU preflight.
- Run the smallest relevant tests during development, then the complete unit
  suite before handoff.
- When changing model topology, LoRA paths, sequence length, sharding, remat,
  flash attention, optimizer behavior, or checkpointing, also run the TPU
  preflight and smoke job.
- Report validation precisely. Distinguish source inspection, CPU/Mac tests,
  TPU preflight, JAX compilation, completed optimizer steps, checkpoint writes,
  and merged export; they are not interchangeable evidence.

## Configuration guidance

- Keep reusable defaults in YAML recipes and expose experiment-specific values
  through dotted command-line overrides.
- The product of `model.mesh.shape` must equal the number of visible JAX
  devices, and `axis_names` must have the same rank as `shape`.
- Keep batch and sharding choices compatible with the target topology.
- Increase `dataset.max_length` only after observing HBM use on the target TPU.
  The default 8192 is intentionally below Open-R1's CUDA recipe length.
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
- Do not commit generated datasets, downloaded model weights, checkpoints,
  logs, profiler output, secrets, or merged artifacts.
- Update `README.md`, the recipe, tests, and preflight together when a behavior
  or operator-facing command changes.
- Do not start a full training run, download large artifacts, publish outputs,
  or delete checkpoints without explicit user authorization.
