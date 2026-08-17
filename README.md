# open-r1-tpu

TPU-native supervised reasoning distillation built with JAX, Grain, Orbax, and
[Google Tunix](https://github.com/google/tunix). The pipeline mirrors the SFT
stage in Hugging Face's `open-r1`: it loads conversational reasoning traces,
applies the model's chat template, and trains a causal language model to emit
the complete assistant reasoning trace before a later GRPO stage.

The default recipe trains `Qwen/Qwen3-1.7B-Base` with LoRA on
`open-r1/Mixture-of-Thoughts` using one 32 GiB TPU v6e device. It supervises
assistant tokens only, filters incomplete or overlength reasoning traces,
writes resumable Orbax checkpoints, and exports a merged Hugging Face-style
safetensors directory.

## Quick start on a TPU VM

Run every step on the TPU VM itself, over SSH. Both scripts are re-runnable, so
a failed step can simply be repeated.

**1. Clone the repository.**

```bash
git clone https://github.com/CMBAgents/open-r1-tpu.git
cd open-r1-tpu
```

**2. Build the environment.**

```bash
./scripts/setup_tpu_vm.sh
```

This installs `uv`, the CPython build pinned in `.python-version`, a `.venv`,
and the project with its test extra. It then confirms JAX sees exactly one TPU
and runs the unit suite. Add `--skip-verify` if another job is already holding
the TPU, or `--recreate` to rebuild `.venv` from scratch.

**3. Fill in the private environment file.**

Step 2 writes `~/.open-r1-tpu.env` with every value commented out. Uncomment
and set the ones you need, then load it:

```bash
export GCS_BUCKET=gs://your-bucket        # source bucket for step 4
export WANDB_ENTITY=your-user-or-team     # W&B account or team
export WANDB_PROJECT=your-project         # W&B project
export HF_TOKEN=hf_...                    # only for Hub-sourced runs
```

```bash
source ~/.open-r1-tpu.env
source .venv/bin/activate
```

Keep deployment-specific values here rather than in the recipe. The file lives
outside the repository and is created `chmod 600`, so nothing lands in git or
in shell history.

**4. Copy GCS bucket data.**

```bash
./scripts/copy_gcs_bucket_data.sh
```

This copies the base model and the dataset from `$GCS_BUCKET` into the ignored
`models/` and `data/` directories, then reports what arrived. It is only needed
if you train from bucket data; skip it to pull from the Hub instead. See
[Copying GCS bucket data](#copying-gcs-bucket-data) for the bucket layout it
expects.

**5. Run preflight.**

```bash
python -m open_r1_tpu.check_env \
  model.model_source=local \
  model.model_path=models/Qwen3-1.7B-Base \
  tokenizer.tokenizer_path=models/Qwen3-1.7B-Base
```

Drop the three overrides to validate a Hub-sourced run instead.

**6. Smoke test, then train.**

```bash
./scripts/run_sft_tpu.sh \
  model.model_source=local \
  model.model_path=models/Qwen3-1.7B-Base \
  tokenizer.tokenizer_path=models/Qwen3-1.7B-Base \
  dataset.name=parquet \
  dataset.config=null \
  dataset.data_files='data/Mixture-of-Thoughts/all/*.parquet' \
  training.project_name="${WANDB_PROJECT}" \
  dataset.max_examples=128 \
  training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2
```

Drop the last four overrides for the full run. The first step includes JAX/XLA
compilation and is much slower than the rest.

## TPU VM setup

Use standard CPython 3.13 (the repository default is 3.13.14) on a TPU VM with
one visible device. Do not use the free-threaded `3.13t` build.

`scripts/setup_tpu_vm.sh` covers step 2 above. To do the same by hand:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'

python - <<'PY'
import jax
print(jax.devices())
assert len(jax.devices()) == 1
PY
```

Set `HF_TOKEN` before launch (Tunix's Hugging Face downloader expects a logged-in
session, including for public repositories). Tunix and its TPU JAX dependency
are installed by `pyproject.toml`; no CUDA packages, PyTorch trainer,
Accelerate, DeepSpeed, or GPU vLLM are used by the SFT stage.

Validate the environment on the TPU VM itself before downloading the full model
or starting a training job:

```bash
python -m open_r1_tpu.check_env
```

This initializes JAX and requires the configured mesh device count to consist
entirely of TPUs. It also checks the installed Tunix/Optax APIs, loads the real
Qwen tokenizer, verifies the assistant-only chat-template boundary, and
confirms that Tunix supplies the Qwen3 merged-LoRA exporter.

### Copying GCS bucket data

Tunix's Hugging Face loader still contacts the Hub even when its download
directory already contains weights, so bucket data has to reach the VM's local
disk and the recipe has to select the explicit local loaders.

`scripts/copy_gcs_bucket_data.sh` does the copy. Run it on the VM: the VM's
service account authenticates to the bucket, and the data moves straight from
GCS to local disk without passing through a workstation. The bucket comes from
`$GCS_BUCKET` or `--bucket`, and is never committed:

```bash
./scripts/copy_gcs_bucket_data.sh
./scripts/copy_gcs_bucket_data.sh --bucket gs://another-bucket
```

It expects `models/Qwen3-1.7B-Base` and `datasets/Mixture-of-Thoughts` inside
the bucket, copying them to `models/Qwen3-1.7B-Base` and
`data/Mixture-of-Thoughts` locally. Set `$GCS_MODEL_PREFIX` or
`$GCS_DATA_PREFIX` for a different layout. Afterwards it reports the Parquet
shard count and on-disk sizes, warning rather than failing silently if either
copy looks empty. `gcloud storage rsync` is incremental, so re-running after an
interrupted copy resumes cheaply.

The equivalent by hand:

```bash
gcloud storage rsync \
  gs://your-bucket/models/Qwen3-1.7B-Base \
  models/Qwen3-1.7B-Base --recursive
gcloud storage rsync \
  gs://your-bucket/datasets/Mixture-of-Thoughts \
  data/Mixture-of-Thoughts --recursive
```

For copied model and data, add these local-input overrides to the launcher; the
checked-in mesh, batch, and sequence defaults already target the one-device VM:

```bash
model.model_source=local
model.model_path=models/Qwen3-1.7B-Base
tokenizer.tokenizer_path=models/Qwen3-1.7B-Base
dataset.name=parquet
dataset.config=null
dataset.data_files='data/Mixture-of-Thoughts/all/*.parquet'
```

Orbax checkpoints may use a `gs://` directory directly. Merged export is a
local-filesystem operation; export locally first, then copy the completed
directory to GCS with `gcloud storage rsync --recursive`.

## Smoke test

Start with a short run before allocating a full training job:

```bash
./scripts/run_sft_tpu.sh \
  dataset.max_examples=128 \
  training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2
```

The first step includes JAX/XLA compilation and will be much slower than later
steps.

## Weights & Biases

Training logs Tunix metrics to the `open-r1-tpu` W&B project by default,
including stepped train/eval loss, perplexity, gradient norm, and the full
resolved recipe. Auxiliary JAX compilation and Orbax checkpoint metrics remain
in TensorBoard because they do not consistently carry a logical training step.
Authenticate once on the TPU VM; enter the API key only at the prompt so it is
not stored in shell history or committed to the repository:

```bash
wandb login
export WANDB_ENTITY=your-user-or-team
```

The default run is named `qwen3-1.7b-reasoning-sft` and grouped under
`qwen3-1.7b-reasoning-distillation`. These values can be overridden normally:

```bash
./scripts/run_sft_tpu.sh \
  training.project_name=my-project \
  training.run_name=my-run \
  training.wandb.entity=my-team
```

Set `training.wandb.mode=offline` to keep W&B data local for later syncing, or
`training.wandb.enabled=false` to disable it. To continue the same W&B run when
restoring an Orbax checkpoint, preserve the W&B log directory and set
`WANDB_RUN_ID` to the original run ID; the recipe's `resume: allow` will then
append to that run.

## Full reasoning SFT

```bash
./scripts/run_sft_tpu.sh
```

The recipe is at
[`recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml`](recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml).
Any value can be overridden using Tunix-style dotted arguments, for example:

```bash
./scripts/run_sft_tpu.sh \
  dataset.config=math \
  training.max_steps=1000
```

Important behavior:

- The input must contain a `messages` column whose final turn is an assistant
  response. This matches `Mixture-of-Thoughts`.
- By default the assistant response must contain `<think>` and `</think>`.
- Loss is masked off for system/user/padding tokens. Set
  `dataset.assistant_only_loss=false` to reproduce full-conversation causal
  loss instead. The trainer uses integer-label cross-entropy rather than
  Tunix's vocabulary-sized one-hot default, which avoids a large temporary at
  reasoning-scale sequence lengths.
- Overlength traces are filtered, not truncated, so training never sees a
  severed chain of thought or missing final answer.
- `dataset.batch_size=1` and `dataset.max_length=1024` are the validated
  32 GiB single-device baseline. With eight accumulation steps, the effective
  batch size is eight; increase sequence length only after measuring HBM use
  and how many complete reasoning traces survive length filtering.

## Checkpoints and GRPO handoff

Training writes resumable Tunix/Orbax LoRA checkpoints under:

```text
artifacts/OpenR1-Distill-Qwen3-1.7B/checkpoints
```

At successful completion, Tunix merges the LoRA delta into the original Qwen3
weights and writes a standard safetensors model under:

```text
artifacts/OpenR1-Distill-Qwen3-1.7B/merged
```

The default `export.overwrite=true` replaces that specific merged-output
directory on a repeated run; checkpoint and model-cache directories are guarded
against accidental use as export targets.

Use that merged directory as the initial policy **and** frozen reference model
for GRPO. In a Tunix Qwen3 GRPO recipe, replace the base-model safetensors path
used by `create_model_from_safe_tensors(...)` with this directory. Keeping the
reference fixed at the SFT initialization makes GRPO's KL term constrain policy
updates relative to the distilled model rather than the original base model.

Merged export currently depends on Tunix's model-specific exporter. The default
Qwen3 recipe supports it. When changing model families, set
`export.enabled=false` unless that Tunix params module provides
`save_lora_merged_model_as_safetensors`.

## Tests

```bash
python -m pytest
```

The unit tests cover config overrides, reasoning-tag filtering, overlength
filtering, chat-template boundaries, and the assistant-only loss mask without
requiring a TPU or downloading model weights. These tests are useful for the
host-independent code, but they are not a substitute for `check_env` plus the
four-step smoke run on the target TPU VM.
