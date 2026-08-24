# open-r1-tpu

TPU-native supervised reasoning distillation built with JAX, Grain, Orbax, and
[Google Tunix](https://github.com/google/tunix). The pipeline mirrors the SFT
stage in Hugging Face's `open-r1`: it loads conversational reasoning traces,
applies the model's chat template, and trains a causal language model to emit
the complete assistant reasoning trace before a later GRPO stage.

The default recipe trains `Qwen/Qwen3-1.7B-Base` with LoRA on
`open-r1/Mixture-of-Thoughts` using one 32 GiB TPU v6e device. It supervises
assistant tokens only, filters incomplete reasoning traces and by default
overlength ones, writes resumable Orbax checkpoints, and exports a merged Hugging Face-style
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
uv sync --frozen --extra test

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
./scripts/copy_gcs_bucket_data.sh --dataset smoltalk
```

It reads `models/Qwen3-1.7B-Base` and `datasets/NAME` from the bucket, writing
them to `models/Qwen3-1.7B-Base` and `data/NAME` locally. `NAME` comes from
`--dataset` or `$GCS_DATASET` and defaults to `Mixture-of-Thoughts`; the
instruction-tuning corpus is `smoltalk`. Set `$GCS_MODEL_PREFIX`,
`$GCS_DATA_PREFIX`, or `$GCS_DATA_GLOB` for a different layout. Afterwards it
reports how many Parquet shards the *training glob* matches — not merely how many
were copied — along with on-disk sizes, warning rather than failing silently if
either copy looks empty. `gcloud storage rsync` is incremental, so re-running
after an interrupted copy resumes cheaply.

The equivalent by hand:

```bash
gcloud storage rsync \
  gs://your-bucket/models/Qwen3-1.7B-Base \
  models/Qwen3-1.7B-Base --recursive
gcloud storage rsync \
  gs://your-bucket/datasets/smoltalk \
  data/smoltalk --recursive
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

### Continue text with the local base model

`Qwen3-1.7B-Base` is a pretrained causal language model rather than a
post-trained chat model. To inspect its native next-token behavior, pass raw
text directly to `complete_qwen_tpu.py`. The script adds no system prompt, role
markers, chat template, or conversation history and generates at most 100 new
tokens by default:

```bash
source .venv/bin/activate
python scripts/complete_qwen_tpu.py \
  --model-path models/Qwen3-1.7B-Base \
  "The capital of France is"
```

Omit the quoted prompt for an interactive loop of independent completions. The
first completion compiles the TPU decode path and is slower than later ones.
Override the generation length when needed:

```bash
python scripts/complete_qwen_tpu.py \
  --model-path models/Qwen3-1.7B-Base \
  --max-new-tokens 200
```

Generation is greedy and stops at the model's `<|endoftext|>` token or the
configured token limit. `--max-prompt-length` defaults to 2048. The completion
client deliberately uses ordinary masked attention: the pinned Tunix sampler
left-pads fixed-length prompts but does not pass the segment IDs that Qwen's
splash-attention path needs to exclude padding. This is slower than splash
attention, but prevents pad/EOS embeddings from changing the continuation.

`chat_qwen_tpu.py` remains available for inspecting role-formatted prompts, but
it applies the tokenizer's system/user/assistant chat template. Base weights
are not expected to follow that template reliably; use a post-trained Qwen
checkpoint for conversational behavior.

### Chat with a training run's own weights

Training writes LoRA adapters alone, so a checkpoint is not a model you can
load on its own. Pass the recipe and the adapters from its latest checkpoint
are restored on top of the base weights:

```bash
python scripts/chat_qwen_tpu.py \
  --recipe recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml
```

The rank, alpha and target modules come from that recipe rather than from
flags, because adapters restored under a different LoRA geometry than they
were trained with produce confident nonsense rather than an error. Add `--step`
to pick an earlier checkpoint, or `--checkpoint-dir` if the artifacts moved.

The restored step is printed on load, and it is rarely the step the run
stopped on: checkpoints are written every
`training.checkpointing_options.save_interval_steps` and only `max_to_keep` of
them survive, so a run killed at step 1744 leaves 1500 as its latest. Asking
for a step that was never written names the ones that were.

Once a run finishes it exports merged weights, which need no recipe:

```bash
python scripts/chat_qwen_tpu.py --model-path artifacts/Qwen3-1.7B-Instruct/merged
```

**One TPU, one process.** The v6e-1 chip is held by whichever process claims it
first, so this cannot run beside a training job. Stop the run, or wait for it.

Two details of the chat script exist to match what the corpus actually teaches,
and both would otherwise be silent:

- **The reply stops at `<|im_end|>`.** Qwen3's template ends every turn with
  that token, but `Qwen3-1.7B-Base` names `<|endoftext|>` as its EOS, and the
  sampler stops at the tokenizer's EOS unless told otherwise. Left alone the
  model runs past the end of its reply and writes your next turn for you.
- **An empty `<think></think>` block is hidden.** The template opens an
  assistant turn with `<think>\n\n</think>\n\n` whenever the message carries
  no reasoning trace, so a corpus without traces teaches the model to emit that
  scaffold before every answer. It is stripped for display; a trace with actual
  content is left alone.

The system prompt defaults to empty, matching `dataset.system_prompt: null` in
the recipes. Pass `--system-prompt` to try one.

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

Send smoke output somewhere disposable. Training restores the newest checkpoint
in `training.checkpoint_dir` automatically, so smoke checkpoints left in
`artifacts/` will be picked up by the next full run and resume it from four
steps of a throwaway configuration:

```bash
  training.checkpoint_dir=/tmp/sft-smoke/checkpoints \
  training.transcripts.output_path=/tmp/sft-smoke/transcripts.jsonl \
  training.wandb.enabled=false
```

## Watching how training is going

Two signals, one quantitative and one qualitative.

**Held-out loss.** `dataset.eval_fraction: 0.01` holds out a slice of the
corpus, and `dataset.eval_max_examples: 64` caps it. The cap matters: a
hundredth of a corpus this size is thousands of examples, and the trainer walks
the whole eval set at every evaluation, so an uncapped split would spend longer
evaluating than training. Evaluation runs every `training.eval_every_n_steps`
and reports `eval/loss` alongside `train/loss`.

**Free-running transcripts.** Teacher-forced loss says nothing about what the
model does when it generates unaided. It cannot tell you whether the model
closes its `<think>` trace, stops at EOS, or loops — every token it scored was
conditioned on ground truth. Sampling a fixed prompt set at a fixed interval
shows exactly that:

```bash
./scripts/run_sft_tpu.sh training.transcripts.enabled=true
```

Transcripts are **disabled by default** because they are not free. Unlike GRPO,
where rollouts are the training signal and already exist, SFT never generates,
so this adds an autoregressive decode, a second XLA compilation, and a KV cache
to a memory profile validated without them. Enable it once you know you have HBM
headroom, and watch the first sampling step for an OOM.

Each interval writes one JSON object per prompt to
`artifacts/OpenR1-Distill-Qwen3-1.7B/transcripts.jsonl`, recording the step, the
prompt, the completion, and flags for whether the reasoning trace was closed and
whether the token budget was exhausted. That last flag matters when reading the
output: a completion that used its whole budget was probably cut off, so a
missing `</think>` there is inconclusive rather than a real failure. The same
records go to W&B as a table under `samples/transcripts` unless
`training.transcripts.log_to_wandb=false`.

Sampling never ends a run. If it OOMs or fails to compile, it logs a warning,
disables itself for the remainder of the run, and training continues.

Tunable values:

```bash
./scripts/run_sft_tpu.sh \
  training.transcripts.enabled=true \
  training.transcripts.every_n_steps=250 \
  training.transcripts.max_new_tokens=512 \
  training.transcripts.temperature=0.7 \
  training.transcripts.prompts='[Prove that sqrt(2) is irrational.]'
```

Greedy decoding (`temperature: 0.0`) is the default so successive samples stay
comparable across steps.

Prompts are padded to `max_prompt_length` before prefill, which defaults to
`model.flash_attention_block_size`. This is not optional padding: the splash
attention kernel requires its block size to divide the prompt length, and left
to itself the sampler pads short prompts to the next power of two, which fails
with `q_block_size=1024 should divide q_seq_len=128`. `cache_size` then defaults
to `max_prompt_length + max_new_tokens`, since the sampler budgets both. If you
disable flash attention, prompt padding reverts to the sampler's own choice.

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

## Instruction tuning before reasoning SFT

[`recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml`](recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml)
trains general instruction following on `smoltalk` (1,043,917 train rows), the
SFT mix that built SmolLM2-1.7B-Instruct. It produces a base model that answers
ordinary requests and stops on `<|im_end|>`, which the reasoning recipe can then
specialise. Both stages use the tokenizer's own Qwen3 chat template, so the turn
structure and end-of-turn token stay the same across them.

Measured under the recipe's own encoder on a stratified 8,000-row sample, the
corpus is short: median templated length 590 tokens, p90 1,677, p99 3,022. The
recipe's `max_length: 2048` therefore retains 95.9% of it, with 13.5% of each
padded sequence carrying gradient. Wider windows buy almost no extra data — 4096
retains 99.6%, 8192 retains 99.8% — while costing more than the token count
suggests, because attention is quadratic: one 8192 sequence runs to roughly 5.5x
one at 2048. The 8192 sequence length on the dataset card describes what its
authors could afford on a multi-GPU node, not a length this corpus needs.

Padding would otherwise dominate at any window — 86.5% of each sequence is pad
even at 2048, and pad positions cost full forward and backward compute because
the masks gate the loss, not the matmuls — so the recipe packs whole examples
into 2048-token windows first fit (`dataset.packing: true`). Attention cannot
cross example boundaries: per-token `segment_ids` gate the splash kernel, the
non-flash path receives a block-diagonal causal mask, and RoPE positions restart
per example. A segment's first token is never supervised, since the causal shift
would predict it from the preceding example. The `segment_ids` gap that still
stands is the sampler's: interactive inference does not pass them, which is why
chat runs with masked attention.

Unlike the reasoning recipe, this one defaults to Parquet already staged on the
VM's local disk rather than to the Hub, so stage the corpus first:

```bash
./scripts/copy_gcs_bucket_data.sh --dataset smoltalk
```

That writes `data/smoltalk/data/all/train-0000{0..8}-of-00009.parquet`, which is
what the recipe globs. `all` is the full mix; the sibling directories under
`data/` are its component subsets. `test-00000-of-00001.parquet` lands beside the
train shards and is deliberately excluded: `dataset.data_files` carries no split
mapping, so every match would be loaded into the train split. The held-out set
comes from `dataset.eval_fraction` instead.

`run_sft_tpu.sh` defaults to the reasoning recipe; select this one with `RECIPE`:

```bash
RECIPE=recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml \
  ./scripts/run_sft_tpu.sh \
  model.model_source=local \
  model.model_path=models/Qwen3-1.7B-Base \
  tokenizer.tokenizer_path=models/Qwen3-1.7B-Base \
  training.project_name="${WANDB_PROJECT}" \
  2>&1 | tee -a artifacts/instruct.log
```

To pull the corpus from the Hub instead, override `dataset.name`, name the
config, and clear the glob:
`dataset.name=HuggingFaceTB/smoltalk dataset.config=all dataset.data_files=null`.

Two settings differ from the reasoning recipe for reasons that are easy to get
wrong. `dataset.require_reasoning_tags` is `false`, because SmolTalk carries no
`<think>`/`</think>` traces and requiring them filters every example into an
empty dataset rather than raising. `dataset.system_prompt` is `null`, because
`prepare_messages` injects a system prompt into any example that lacks one, and a
reasoning instruction in front of responses that ignore it teaches the model to
disregard its system prompt.

The stage writes a merged export to `artifacts/Qwen3-1.7B-Instruct/merged`. Point
the reasoning run at it to chain the two:

```bash
./scripts/run_sft_tpu.sh \
  model.model_source=local \
  model.model_path=artifacts/Qwen3-1.7B-Instruct/merged
```

Reasoning SFT on its own data will erode general instruction following, so mix
some of this stage's corpus back in if both behaviours matter. Going straight
from base to reasoning SFT is also viable — the DeepSeek-R1 distills did exactly
that — and sequencing is mainly worth it to isolate what the reasoning stage adds.

## Full reasoning SFT

### The complete run, from bucket data with transcripts

The default recipe runs 5000 optimizer steps. At `gradient_accumulation_steps: 8`
and `batch_size: 1` that consumes 40,000 examples — well under one epoch of
`Mixture-of-Thoughts`, whose `all` subset holds roughly 349k rows before length
filtering. Along the way it writes 20 checkpoints (keeping the newest 2),
evaluates 11 times (a baseline plus every 500 steps), and samples transcripts 10
times.

Training runs for hours, so start it under `tmux` and it will survive an SSH
drop:

```bash
tmux new -s sft
```

Inside the session:

```bash
cd ~/open-r1-tpu
source ~/.open-r1-tpu.env
source .venv/bin/activate

LOCAL_INPUTS=(
  model.model_source=local
  model.model_path=models/Qwen3-1.7B-Base
  tokenizer.tokenizer_path=models/Qwen3-1.7B-Base
  dataset.name=parquet
  dataset.config=null
  dataset.data_files='data/Mixture-of-Thoughts/all/*.parquet'
)

mkdir -p artifacts
./scripts/run_sft_tpu.sh "${LOCAL_INPUTS[@]}" \
  training.project_name="${WANDB_PROJECT}" \
  training.transcripts.enabled=true \
  2>&1 | tee -a artifacts/train.log
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t sft`. Drop the
`LOCAL_INPUTS` expansion to pull the model and dataset from the Hub instead,
which needs `HF_TOKEN` set.

An evaluation runs *before* the first training step. That is a baseline reading,
not a misconfigured interval: the trainer evaluates unconditionally at startup
whenever a held-out split exists, independently of
`training.eval_every_n_steps`.

### Resuming, intended and unintended

Training restores the newest checkpoint found in `training.checkpoint_dir`
without being asked. To continue an interrupted run, leave the checkpoints in
place, relaunch the identical command, and set `WANDB_RUN_ID` to the original
run so the charts stay continuous.

The same behaviour bites when the checkpoints came from a *different*
configuration — a smoke run, say. The startup log tells you which it is:

```text
Found 2 checkpoint steps in .../checkpoints
Restored params from step: 4
```

If that step is not where you meant to resume, stop, move the directory aside,
and relaunch:

```bash
mv artifacts/OpenR1-Distill-Qwen3-1.7B/checkpoints \
   artifacts/stale-checkpoints-$(date +%s)
```

### Watching a run in progress

```bash
tail -f artifacts/train.log
```

Orbax's per-step bookkeeping is demoted to `DEBUG`, not discarded. Tunix calls
`CheckpointManager.save` on every optimizer step and lets Orbax's save policy
decide whether to write, and Orbax rebuilds its handler registry before
reaching that decision, so six lines about `BasePyTreeCheckpointHandler`,
`DefaultCheckpointHandlerRegistry` and `barrier_sync_fn` surround each step's
single loss line whether or not anything is saved. Every library here logs
through absl's one logger at `INFO`, so neither the level nor the logger name
separates them until they are relabelled. Warnings and errors keep their
level.

```bash
./scripts/run_sft_tpu.sh --log-level debug    # the demoted records, as DEBUG
./scripts/run_sft_tpu.sh --log-level warning  # problems only
```

This lives in [`src/open_r1_tpu/logging.py`](src/open_r1_tpu/logging.py).
Quietening another package that hides behind absl is one entry in
`NOISY_PACKAGES`; a library with a logger of its own needs no help, since its
level can simply be set.

Transcripts accumulate as JSON lines, one object per prompt per interval. This
summarizes whether the model is learning to close its reasoning traces:

```bash
python - <<'PY'
import collections, json

by_step = collections.defaultdict(list)
with open("artifacts/OpenR1-Distill-Qwen3-1.7B/transcripts.jsonl") as handle:
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

`closed` should climb toward the prompt count over the first few thousand steps.
If it stays at zero while `train/loss` falls, that is the failure teacher-forced
loss cannot show, and it is worth stopping for.

### Overriding the recipe

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
  response. This matches `Mixture-of-Thoughts`. A corpus that names things
  differently — ShareGPT's `conversations` of `{from, value}` with `human`/`gpt`
  roles — is adapted by `dataset.messages_column` and `dataset.message_schema`
  in the recipe, not by code.
- By default the assistant response must contain `<think>` and `</think>`.
- Loss is masked off for system/user/padding tokens. Set
  `dataset.assistant_only_loss=false` to reproduce full-conversation causal
  loss instead. The trainer uses integer-label cross-entropy rather than
  Tunix's vocabulary-sized one-hot default, which avoids a large temporary at
  reasoning-scale sequence lengths.
- Overlength traces are filtered rather than truncated
  (`dataset.overlength_policy: drop`, the default), so training never sees a
  severed chain of thought or missing final answer. `truncate` cuts the render
  on the right instead, keeping the prompt and leaving the sequence
  deliberately unterminated; it is for corpora whose traces are themselves
  incomplete, where dropping would exclude most of the data. See
  [Distilling from OpenThoughts3 under truncation](#distilling-from-openthoughts3-under-truncation).
- `dataset.batch_size=1` and `dataset.max_length=1024` are the validated
  32 GiB single-device baseline. With eight accumulation steps, the effective
  batch size is eight; increase sequence length only after measuring HBM use
  and how many complete reasoning traces survive length filtering.

## Distilling from OpenThoughts3 under truncation

[`recipes/Qwen3-1.7B-OT3/sft/config_distill.yaml`](recipes/Qwen3-1.7B-OT3/sft/config_distill.yaml)
trains the same model on `open-thoughts/OpenThoughts3-1.2M` at
`dataset.max_length: 16384`. It is the one recipe that truncates rather than
drops:

```bash
RECIPE=recipes/Qwen3-1.7B-OT3/sft/config_distill.yaml ./scripts/run_sft_tpu.sh
```

The corpus forces that choice. Its traces were generated by QwQ-32B under a
context budget of roughly 16,800 tokens, so most of them stop mid-sentence:
`<think>` opens in 100% of rows, but 62% carry no closing `</think>`. Dropping
incomplete or overlength traces here excludes most of the corpus instead of
filtering bad data, which is why `dataset.require_reasoning_tags` is `false` and
`dataset.overlength_policy` is `truncate`. The dataset's authors ran the same
comparison, reported that filtering incomplete traces *hurts* downstream
performance, and trained on the truncated majority; their trainer truncates at
`cutoff_len` by default, so their overlength rows were cut a second time during
training.

Measured on a 21,000-row sample with the Qwen3-1.7B-Base tokenizer, rendered as
ChatML with the recipe's system prompt and reweighted to the corpus's true
850k/250k/100k domain mix: truncation keeps ~100% of the 1.2M rows and holds the
domain mix at the corpus's 71% math / 21% code / 8% science, while dropping at
the same 16384 would keep only ~31.7% (~381k rows) and skew it to 58% math /
23% science / 19% code. About 68% of rows exceed 16384 and are cut.

Two consequences follow. Packing gains largely collapse: an example cut at
`max_length` fills its window alone, so only the shorter, science-heavy tail
still packs. And an epoch is 1.2M examples rather than ~381k, so
`training.max_steps: 4000` at `gradient_accumulation_steps: 32` covers ~10.7% of
one epoch — a first pass in which no example is seen twice.

Rows are ShareGPT-style, a `conversations` list of `{from, value}` with
`human`/`gpt` roles, which `dataset.message_schema` maps onto `role`/`content`.
Without that block every row fails message validation and is filtered, leaving
an empty dataset with no error to read: check the retained count on a small
`dataset.max_examples` run before starting a long one.

Nothing about this recipe's memory profile has been measured. The 32 GiB
baseline was smoke-tested at `max_length: 1024`; this asks for 16x the sequence,
and under truncation most examples are full-length, so the worst case is now the
common case. Measure peak HBM in a short run before committing TPU time, and
revisit `model.remat_config` first.

## Math reasoning SFT on OpenR1-Math-220k

[`recipes/Qwen3-1.7B-Math/sft/config_distill.yaml`](recipes/Qwen3-1.7B-Math/sft/config_distill.yaml)
is the short-window counterpart to the OpenThoughts3 recipe: a full-parameter
fine-tune of `Qwen3-1.7B-Base` on `open-r1/OpenR1-Math-220k` at
`dataset.max_length: 4096`, dropping over-length traces rather than truncating
them.

```bash
./scripts/copy_gcs_bucket_data.sh --dataset OpenR1-Math-220k
RECIPE=recipes/Qwen3-1.7B-Math/sft/config_distill.yaml ./scripts/run_sft_tpu.sh
```

The corpus permits what OpenThoughts3 does not. Measured on 3,000 rows sampled
evenly across the 93,733 in the `default` config, rendered with the
Qwen3-1.7B-Base chat template and the recipe's system prompt: `<think>` appears
in 100% of assistant messages and `</think>` in 99.9%, and the templated length
has a median of 4,855 tokens (p75 8,127, p90 12,070). Retention under `drop` is
11.6% at 2048, **40.5% at 4096** (~37,900 rows), 75.4% at 8192 and 98.5% at
16384.

4096 therefore buys complete traces rather than more of them. Every retained
example carries a full chain of thought ending in a boxed answer, which is the
property a reasoning stage exists to teach and the one truncation destroys; the
price is the 59.5% of the corpus that does not fit. Retained examples average
2,609 tokens, 63.7% of the window, so `dataset.packing` is worth keeping on and
windows carry roughly 1.5 examples each.

The run is a full fine-tune rather than LoRA because a reasoning stage is
defined by emitting `<think>`, `</think>` and `<|im_end|>`, whose rows live in
the tied embedding matrix that LoRA freezes — the failure the instruct stage
already hit. It trains the base model directly rather than the instruct stage's
merged export, following Olmo 3, DeepSeek-R1 and Qwen3, all of which run
long-CoT SFT on the base model. The consequence is that the output is a math
reasoner, not a general chat model: this corpus carries no instruction-following
or open-ended chat data.

`training.max_steps: 2400` is about three epochs — ~99M retained tokens pack into
~25,400 windows, or ~790 optimizer steps per epoch at 32 windows per step. That
makes held-out loss a real overfitting signal here, unlike the OpenThoughts3 run,
which sees no example twice. It is still `eval_fraction: 0.0` by default, because
the instruct run's 2048-token evals spiked HBM to 31.23 of 31.25 GiB and a full
fine-tune at twice that sequence has less room. Peak HBM at 4096 is unmeasured:
measure it in a short run before committing TPU time, then enable eval.

## Benchmark evaluation

Held-out loss is teacher-forced: every scored token is conditioned on ground
truth, so it cannot say whether the model closes its reasoning trace, stops, or
reaches the right answer unaided. Those are the questions a reasoning stage is
judged on, and only free generation scored against a reference answers them.

The stack is three decoupled layers. Generation is vLLM on the TPU, serving the
merged export behind an OpenAI-compatible endpoint. The harness is LightEval,
reached over HTTP through its litellm backend. Scoring is whatever metric the
LightEval task declares, which for maths is symbolic equivalence via
latex2sympy2-extended rather than string equality. `src/open_r1_tpu/evaluate.py` owns the layers either
side of the harness — it validates the recipe, runs the harness once per seed,
and reduces what the harness wrote into a single summary — and imports neither
JAX, Tunix, nor vLLM.

### Installing and running

Install the host harness and pull the inference service in one re-runnable
step:

```bash
./scripts/setup_tpu_vm.sh --with-eval
```

This uses the tested uv 0.12.5 installer and
`uv sync --frozen --extra eval --extra test`, so the Python version, direct
evaluation dependencies, and complete transitive environment come from
`.python-version`, `pyproject.toml`, and the committed `uv.lock`. It then pulls
the official vLLM TPU 0.27.0 image by immutable registry digest. A missing or
mismatched host dependency is an evaluation preflight error rather than a
warning recorded after an expensive run.

**vLLM is not installed on the host.** Nothing in this package imports it: the
launcher starts the pinned container and everything after that is HTTP. The
container wrapper supplies the TPU requirements (`--privileged`, host
networking, and shared memory), bind-mounts the selected merged export read-only,
and persists Hugging Face and vLLM/XLA caches in the
`open-r1-tpu-huggingface-cache` and `open-r1-tpu-vllm-cache` Docker volumes.
It forwards `HF_TOKEN` by environment-variable name only when present; token
values never enter recipes, printed commands, or summaries.

The recipes record both the readable vLLM release tag and its immutable digest:

```yaml
server:
  serve_command: ["scripts/run_vllm_tpu_container.sh"]
  image: "docker.io/vllm/vllm-tpu:v0.27.0@sha256:d6748bc7b1b020ab6411506d4bf30f8bfabb5db2b8505328f26d1a545b479df8"
```

`scripts/run_vllm_tpu_container.sh` automatically uses direct Docker access or
passwordless `sudo docker`. The latter is the default on fresh TPU VMs where
the login user is not in the Docker group. It keeps the container in the
foreground, records its CID, and explicitly stops it on interrupts so the TPU
is not left held by an orphan.

An external Python 3.12 vLLM environment remains available as an escape hatch;
disable the container image when overriding the command:

```yaml
server:
  serve_command: ["/opt/vllm-venv/bin/vllm", "serve"]
  image: null
```

That environment is deliberately reported as unchecked by preflight, because
its transitive packages are no longer governed by this repository's lock or
the recorded image digest.

Evaluation runs *after* training rather than beside it. Only one process can
hold the TPU chip, so stop the training job before starting the server.

### Preflight, then the smoke tier

```bash
python -m open_r1_tpu.check_eval_env \
  --config recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml
```

This checks the exact host dependency versions, Docker access and the pinned
local image, the export it is pointed at, and the recipe's task names. Failures
are caught before vLLM claims the TPU. A merged
export missing its tokenizer files or its chat template loads far enough to
serve requests and then answers off-distribution, producing a number that
measures the wrong thing. And Qwen3-Base names `<|endoftext|>` as its EOS while
the chat template closes turns with `<|im_end|>`, so a server left to the
tokenizer's own EOS runs past the end of every reply and writes the user's next
turn as well — which under a benchmark reads as a model that cannot stop
reasoning. The recipes set `sampling.stop` to `<|im_end|>` for that reason.
Third, LightEval moves tasks between suites and releases, so a recipe naming one
that no longer exists is worth hearing about now rather than after the server has
spent fifteen minutes loading weights. A name that exists in a different suite is
reported with the suite it actually lives in.

```bash
RECIPE=recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

`scripts/run_eval_tpu.sh` owns the evaluation-level vLLM lifecycle: it builds
the container command from the recipe, waits for the server to answer, runs the
evaluation, and terminates its process group on the way out. The container
wrapper owns the Docker-level stop/removal. Both halves read the same recipe
and dotted overrides, so the image, port, model name, and context window cannot
drift apart. Evaluation summaries record the host stack, immutable image, and
complete constructed server command. Set `SKIP_SERVER=1` to reuse a server that
is already up.

To update the evaluation environment intentionally, change the exact versions
in `pyproject.toml` and `src/open_r1_tpu/evaluation_stack.py`, run `uv lock`,
then repeat the unit suite and TPU smoke evaluation. Updating vLLM likewise
requires a new release tag and registry digest plus a real TPU smoke run; never
replace either pin with `latest`.

### vLLM versus Tunix generation speed

An evaluation's seconds per sample do not establish which inference engine is
faster. LightEval currently submits one request at a time, leaving vLLM's
continuous batching idle, while Tunix compiles a static batch shape and samples
it directly in-process. Run the controlled comparison on the TPU instead:

```bash
./scripts/benchmark_generation_tpu.sh
```

The launcher serves the merged export with vLLM, measures it, releases the TPU,
then loads the same export into Tunix's direct `Sampler` and measures that. The
default workload covers batch/concurrency 1 (the current LightEval path) and 8,
using the same 16 distinct, pre-rendered prompts, two measured repetitions,
greedy decoding, and 128 forced output tokens. One warm-up batch per shape is
excluded from steady-state throughput; server/model startup and warm-up times
are retained separately. Tunix flash attention is disabled for this short-prompt
inference workload, avoiding its 1024-token splash block padding requirement.

Raw results land in `artifacts/Qwen3-1.7B-Math/generation-speed/{vllm,tunix}.json`.
`comparison.json` and `comparison.md` report output tokens per second, samples
per second, and the Tunix/vLLM ratio at each matching batch size. This is a speed
test only: EOS is deliberately ignored so both engines execute exactly the same
number of decode steps. Keep the ordinary evaluation path for termination and
accuracy.

### The tiers

| Recipe | Cost | When |
| --- | --- | --- |
| `eval/tier0_smoke.yaml` | ~5 min | Every run. GSM8K, 200 problems, greedy. |
| `eval/tier1_core.yaml` | ~1 h | Every checkpoint worth keeping. MATH-500, AMC23, GSM8K over three seeds. |
| `eval/tier2_headline.yaml` | hours | Milestones. AIME24, AIME25, OlympiadBench over ten seeds. |
| `eval/tier3_regression.yaml` | hours | Milestones. IFEval, GPQA-Diamond, MMLU-Pro. |

Tier 0 does not measure ability; it catches a model that is broken in a way loss
cannot show. Tier 1 is the tier that decides whether a recipe change helped.
Tier 2 exists because AIME is the number the field quotes, not because 30
problems can settle an argument — one extra correct answer there moves pass@1 by
3.3 points. Tier 3 answers what the math-only corpus cost, which is an open
question for this project: `OpenR1-Math-220k` carries no instruction-following
or chat data at all.

### Seeds are not optional

Seed variance alone moves small reasoning benchmarks by 5–15 points
([arXiv 2504.07086](https://arxiv.org/abs/2504.07086)), which is more than most
recipe changes are worth. Every task runs once per seed and is reported as mean
and standard deviation; `aggregate_across_seeds` reports a null standard
deviation at one seed rather than a reassuring `0.0`. Three seeds is the
documented minimum at MATH-500's size, ten at AIME's.

Published numbers are not a baseline either — they were produced by a different
stack. Measure the base model on this one:

```bash
RECIPE=recipes/Qwen3-1.7B-Math/eval/tier1_core.yaml ./scripts/run_eval_tpu.sh \
  server.model_path=models/Qwen3-1.7B-Base
```

### What the summary records

The run writes `summary_<tier>.json` under `reporting.output_dir`, holding the
per-task metrics aggregated across seeds, the sampling parameters, the model
path, and the installed versions of LightEval, litellm and
latex2sympy2-extended. A result that does not name its stack cannot be compared
with one produced months later.
`reporting.summary_path` accepts a `gs://` URI to land it beside the checkpoint
it scored.

Alongside accuracy it records four things read out of LightEval's detail shards,
each diagnosing a failure accuracy alone cannot separate:

- `truncation_rate` — the fraction that hit the token cap. A truncated trace
  scores as wrong and reads as a reasoning failure, so if this is not near zero
  then `sampling.max_new_tokens` is too low and the accuracy under it is not
  trustworthy.
- `reasoning_closed_rate` and `answer_marker_rate` — whether the model produces
  the shape SFT was teaching, independent of whether the answer is right.
- `mean_completion_tokens` — the length-inflation signal.

`truncation_rate` and `mean_completion_tokens` are `null` when the detail shards
carry no token counts, which is honest about the gap rather than filling it with
a character-length guess.

Set `reporting.wandb.run_id` to the training run's W&B id to put these numbers on
the same run as the loss curves. W&B resumes by id, never by name, so without one
this logs to a standalone run rather than silently appending to whichever run
happens to share a name.

### Known constraints

- **Task names are not verified.** LightEval renames and moves tasks between
  releases, and the names in the recipes have not been run. Confirm them with
  `lighteval tasks list` before the first run; a wrong name fails immediately and
  costs nothing.
- **Only generative tasks work.** An OpenAI-compatible endpoint returns no
  per-token log probabilities for a supplied continuation, so LightEval's
  loglikelihood tasks — plain MMLU, HellaSwag, ARC — cannot run through this
  backend at all. Tier 3 uses generative tasks throughout for that reason.
- **Single-chip vLLM is undocumented.** The vLLM TPU docs recommend v6e as a
  generation but say nothing about `v6e-1`. Nothing here has been run on a TPU.

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
filtering and truncation, message-schema mapping, chat-template boundaries, and
the assistant-only loss mask without
requiring a TPU or downloading model weights. These tests are useful for the
host-independent code, but they are not a substitute for `check_env` plus the
four-step smoke run on the target TPU VM.

## Linting and type checking

Ruff and pyright run as pre-commit hooks. Install them once per clone:

```bash
python -m pip install -e '.[dev]'
pre-commit install
```

To check the whole tree without committing:

```bash
pre-commit run --all-files
```

Pyright reports the TPU-only imports as warnings off target, so the same checks
pass on a laptop and on the TPU VM.
