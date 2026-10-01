# Training

SFT (`python -m open_r1_tpu.sft.run`) and GRPO
(`python -m open_r1_tpu.grpo.run`) share the recipe loader, checkpointing,
export and logging described here. The README's quick start is the short
path; this page covers the rest.

## Environment

`scripts/setup_tpu_vm.sh` sets up a TPU VM. By hand, with
[uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --frozen --extra test
source .venv/bin/activate
python -c "import jax; print(jax.devices())"
```

This installs the CPython 3.13 in `.python-version` (not the free-threaded
`3.13t` build), Tunix and TPU JAX. Set `HF_TOKEN`: Tunix's Hugging Face
downloader needs it even for public repositories.

Before a training job, run the preflight on the VM:

```bash
python -m open_r1_tpu.sft.preflight --config "$RECIPE"
```

It checks that JAX sees as many TPU chips as `model.mesh.shape` needs, that the
installed Tunix and Optax have the APIs the trainer uses, that the real
tokenizer gives a clean assistant boundary, and that the recipe's merged
export is supported for its model.

## Data from a GCS bucket

Tunix's Hugging Face loader contacts the Hub even when the weights are already
on disk, so data kept in a bucket is copied to local disk and loaded with the
recipe's local loaders. Run the copy on the VM, which authenticates with its
service account; a re-run resumes an interrupted copy:

```bash
gcloud storage rsync --recursive gs://your-bucket/Qwen2.5-Math-1.5B-RoPE-300k \
  models/Qwen2.5-Math-1.5B-RoPE-300k
gcloud storage rsync --recursive gs://your-bucket/OpenR1-Math-220k \
  data/OpenR1-Math-220k
```

Orbax checkpoints can go straight to a `gs://` directory. Merged export must
go to local disk; copy the finished directory to GCS afterwards with
`gcloud storage rsync --recursive`.

## Smoke test

Run a few steps before a full job. With `RECIPE` set:

```bash
python -m open_r1_tpu.sft.run --config "$RECIPE" \
  dataset.max_examples=128 training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2 \
  training.checkpoint_dir=/tmp/sft-smoke/checkpoints \
  training.metrics_log_dir=/tmp/sft-smoke/logs \
  training.transcripts.output_path=/tmp/sft-smoke/transcripts.jsonl \
  training.wandb.enabled=false export.enabled=false
```

Keep smoke output out of `artifacts/`: training resumes from the newest
checkpoint in `training.checkpoint_dir`, so a full run would continue from the
smoke run's four steps. The first step includes XLA compilation and is much
slower than the rest.

## Long runs

Start long runs under `tmux` so they survive an SSH drop:

```bash
tmux new -s sft
source ~/.open-r1-tpu.env && source .venv/bin/activate
export RECIPE=recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
mkdir -p artifacts
python -m open_r1_tpu.sft.run --config "$RECIPE" \
  training.project_name="${WANDB_PROJECT}" 2>&1 | tee -a artifacts/train.log
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t sft`. To stop a run,
interrupt the Python process (`Ctrl-c` in the pane, or kill its PID): killing
only the tmux session can leave it holding the TPU.

**Resuming.** Training restores the newest checkpoint in
`training.checkpoint_dir` without being asked. To continue an interrupted run,
relaunch the same command, with `WANDB_RUN_ID` set to the original run's id so
W&B appends to it. The startup log shows what was restored:

```text
Restored params from step: 4 in 12.345 seconds
```

`--log-level debug` also shows Orbax's list of the checkpoint steps it found.

If that is not the run you meant to continue, stop, move the checkpoint
directory aside, and relaunch.

**Logs.** Tunix asks Orbax to save on every step, and Orbax logs several lines
each time whether or not it writes. These are demoted to `DEBUG`
([core/logging.py](../src/open_r1_tpu/core/logging.py), `NOISY_PACKAGES`), so
the log shows one line per step. Pass `--log-level debug` to see them, or
`--log-level warning` for problems only.

## Monitoring

**Held-out loss.** `dataset.eval_fraction` holds out part of the corpus and
`dataset.eval_max_examples` caps it; without the cap, evaluating a hundredth of
a large corpus takes longer than training. `eval/loss` is reported every
`training.eval_every_n_steps`, and once before the first step.

**Transcripts.** Teacher-forced loss cannot show whether the model closes its
`<think>` block, stops, or loops when generating on its own. Transcripts
sample a fixed set of prompts at intervals:

```bash
python -m open_r1_tpu.sft.run --config "$RECIPE" \
  training.transcripts.enabled=true \
  training.transcripts.every_n_steps=250 \
  training.transcripts.max_new_tokens=512 \
  training.transcripts.prompts='[Prove that sqrt(2) is irrational.]'
```

They are off by default: sampling adds a decode compilation and a KV cache to
the training memory, so enable them only with HBM to spare. If sampling fails,
it logs a warning and turns itself off; training continues. Decoding is greedy
by default, so samples compare across steps. Prompts are padded to
`max_prompt_length`, by default the flash-attention block size, because the
splash kernel needs its block size to divide the prompt length.

Each sample is a JSON line in `training.transcripts.output_path` and a row in
the W&B table `samples/transcripts`, with the step, prompt, completion,
`reasoning_balanced` (the `<think>` block was closed) and `hit_token_cap` (the
budget ran out, so an open block may just be cut off). To see whether the
model is learning to close its reasoning:

```bash
python - <<'PY'
import collections, json

by_step = collections.defaultdict(list)
with open("artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/transcripts.jsonl") as handle:
    for line in handle:
        record = json.loads(line)
        by_step[record["step"]].append(record)

for step in sorted(by_step):
    rows = by_step[step]
    closed = sum(row["reasoning_balanced"] for row in rows)
    capped = sum(row["hit_token_cap"] for row in rows)
    print(f"step {step:>6}  closed {closed}/{len(rows)}  hit cap {capped}")
PY
```

`closed` should climb towards the prompt count over the first few thousand
steps. If it stays at zero while the loss falls, stop and investigate.

**Weights & Biases.** Training logs stepped train and eval loss, perplexity,
gradient norm and the resolved recipe to W&B, in the project
`training.project_name` (default `open-r1-tpu`). JAX compilation and Orbax
metrics go to TensorBoard only, because they carry no training step. Run
`wandb login` once, set `WANDB_ENTITY` or `training.wandb.entity`, and
override the run with `training.run_name` and `training.wandb.group`. Set
`training.wandb.mode=offline` to sync later, or `training.wandb.enabled=false`
to turn it off.

## Recipes

Any recipe value can be overridden with a dotted argument, such as
`training.max_steps=1000`. Overrides apply before validation. Recipes reject
unknown keys and suggest the closest match, so a typo fails at load time. The
checked sections are `dataset`, `optimizer`, `training` (with
`training.wandb` and `training.transcripts`), `export`, and for GRPO `grpo`
and `rollout`; `model`, `tokenizer` and `training.checkpointing_options` pass
through to Tunix. A recipe can `extends:` one base recipe, by a path relative
to itself.

SFT data:

- Each example needs a `messages` column whose last turn is the assistant's.
  Corpora laid out differently, such as ShareGPT's `conversations` of
  `{from, value}`, are mapped with `dataset.messages_column` and
  `dataset.message_schema`.
- By default the assistant turn must contain `<think>` and `</think>`.
- Loss covers only the assistant's tokens; `dataset.assistant_only_loss=false`
  trains on the whole conversation.
- Examples longer than `dataset.max_length` are dropped, so training never
  sees a cut-off trace. `dataset.overlength_policy: truncate` instead cuts
  them and leaves them unterminated, for corpora whose traces are mostly
  cut off already.
- `dataset.packing: true` packs whole examples into each window. Attention
  never crosses example boundaries, and positions restart per example.
- Raise `dataset.max_length` only after measuring peak HBM, and check how many
  examples survive the length filter.

## Checkpoints and export

Training writes resumable Orbax checkpoints to `training.checkpoint_dir`. When
it finishes, it writes a Hugging Face safetensors model to
`export.output_dir`: the trained weights of a full fine-tune, or a LoRA run's
adapter merged into the base. With `export.overwrite: true`, which every
shipped recipe sets, a repeated run replaces that directory; unset, an
existing directory is refused. The checkpoint, model-cache and repository
directories are never accepted as export targets.

To start GRPO from an SFT export, set the GRPO recipe's `model.model_path` to
it, and `model.rope_theta` to the SFT recipe's value, since the loader reads
the recipe rather than the export's `config.json`. GRPO loads the export
twice: as the policy, with a new LoRA adapter, and as the frozen reference
whose KL term keeps the policy close to it.

Full fine-tune export supports Qwen2 and Qwen3. Merged LoRA export uses
Tunix's own exporter where the model has one (Qwen3, for example), and this
project's for Qwen2. For other models, set `export.enabled=false`. The SFT
preflight reports an unsupported export, or one with no local base model,
before training starts.

To export a checkpoint from the middle of an SFT run:

```bash
python scripts/export_checkpoint.py \
  --recipe recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml \
  --step 2000 --output artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/step-2000/merged
```
