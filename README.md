# open-r1-tpu

`open-r1-tpu` is a TPU-native post-training and evaluation workflow for open
language models. It supports instruction tuning, reasoning distillation and
GRPO with JAX and [Google Tunix](https://github.com/google/tunix).

The repository is organized around complete, repeatable model-development
runs: train models, save resumable Orbax checkpoints, export merged
safetensors weights, and evaluate the resulting model against versioned,
tiered benchmark recipes. Evaluation uses LightEval metrics and fixed task
definitions; self-hosted [Langfuse](https://langfuse.com/) can optionally
record each task-and-seed run with its inputs, outputs, traces and scores.

It includes:

- TPU-focused recipes for reasoning SFT and GRPO;
- reproducible local or GCS-backed data and model staging;
- checkpointing, model export, and W&B training metrics;
- multi-seed benchmark evaluation and consensus scoring; and
- optional Langfuse tracing and experiment review.

Deployment-specific values and credentials stay outside the repository; recipes
and scripts remain portable across TPU VMs.

## Package layout

The Python package is grouped by workflow rather than kept as one flat module
directory:

- `open_r1_tpu.core` contains configuration parsing and shared logging;
- `open_r1_tpu.model` contains model creation, optimizer, metrics logging,
  tokenizing helpers, and checkpoint/export handling shared by the SFT and
  GRPO training stages;
- `open_r1_tpu.sft` contains data preparation, SFT orchestration,
  transcripts, and the training preflight;
- `open_r1_tpu.grpo` contains GRPO prompt dataset loading, reward functions,
  and orchestration;
- `open_r1_tpu.evaluation` contains LightEval orchestration, evaluation
  preflight, immutable stack pins, and the vLLM/Tunix speed benchmark.

Shell scripts remain the supported operator-facing launchers and invoke the
corresponding modules inside those subpackages.

## Quick start on a TPU VM

Run every step on a TPU v6e-4 VM, over SSH. Both scripts are re-runnable, so a
failed step can simply be repeated. The example trains
[`recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml`](recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml),
a full fine-tune of Qwen2.5-Math-1.5B on OpenR1-Math-220k; see
[Reasoning distillation](#reasoning-distillation-qwen25-math-15b-on-openr1-math-220k).

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
and the project with its test extra. It then confirms JAX sees the TPU and
runs the unit suite. Add `--with-eval` to also install the evaluation stack,
`--skip-verify` if another job is already holding the TPU, or `--recreate` to
rebuild `.venv` from scratch.

**3. Fill in the private environment file.**

Step 2 writes `~/.open-r1-tpu.env` with every value commented out. Uncomment
and set the ones you need, then load it:

```bash
export HF_TOKEN=hf_...                    # Hub downloads
export WANDB_ENTITY=your-user-or-team     # W&B account or team
export WANDB_PROJECT=your-project         # W&B project
export GCS_BUCKET=gs://your-bucket        # only to stage data from a bucket
```

```bash
source ~/.open-r1-tpu.env
source .venv/bin/activate
```

Keep deployment-specific values here rather than in the recipe. The file lives
outside the repository and is created `chmod 600`, so nothing lands in git or
in shell history.

**4. Stage the base model and the training data.**

```bash
hf download open-r1/OpenR1-Math-220k --repo-type dataset \
  --include 'data/*' --local-dir data/OpenR1-Math-220k
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B-RoPE-300k
python scripts/set_rope.py models/Qwen2.5-Math-1.5B-RoPE-300k \
  --rope-theta 300000 --max-position-embeddings 32768
```

The last command makes the long-context variant of the base the recipe trains
from, by editing its `config.json` only. To copy data you keep in a GCS bucket
instead, see [Copying GCS bucket data](#copying-gcs-bucket-data).

**5. Run preflight.**

```bash
export RECIPE=recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
python -m open_r1_tpu.sft.preflight --config "$RECIPE"
```

**6. Smoke test, then train.**

```bash
./scripts/run_sft_tpu.sh \
  training.project_name="${WANDB_PROJECT}" \
  dataset.max_examples=128 \
  training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2 \
  training.checkpoint_dir=/tmp/sft-smoke/checkpoints \
  export.enabled=false
```

Drop every override but the first for the full run: 6,710 optimizer steps,
about 19 hours, ending with a merged export in
`artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged`. The first step includes
JAX/XLA compilation and is much slower than the rest. `run_sft_tpu.sh` trains
whichever recipe `RECIPE` names; the smoke writes its checkpoints to `/tmp`
because training resumes from the newest checkpoint in
`training.checkpoint_dir`.

**7. Evaluate the export.** With the `--with-eval` environment:

```bash
RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

See [Benchmark evaluation](#benchmark-evaluation).

## Recipes

Recipes live in `recipes/<base model>/<stage>/<dataset>.yaml`, named by the
Hugging Face model the run starts from. Evaluation tiers are
`recipes/<model>/eval/tier<N>_<name>.yaml` and extend that directory's
`base.yaml`. Outputs go under `artifacts/<output model name>/`.

| Recipe | What it does | Hardware | Tested |
| --- | --- | --- | --- |
| `Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml` | Reasoning distillation, full fine-tune | v6e-4 | Yes: MATH-500 83.3% at tier 1 |
| `Qwen2.5-Math-1.5B/eval/` | Tiers 0-3 for that export | 1 chip | Yes |
| `Qwen2.5-Math-1.5B/grpo/dapo-math-17k.yaml` | GRPO on that export | v6e-1 | No |
| `Qwen2.5-1.5B/grpo/simplerl-zoo.yaml` | GRPO positive control against SimpleRL-Zoo | v6e-4 | Yes: GSM8K 68.2, MATH-500 54.7 |
| `Qwen2.5-1.5B/eval/` | SimpleRL-Zoo's tier 1 protocol | 1 chip | Yes |
| `DeepSeek-R1-Distill-Qwen-1.5B/eval/` | The reference model, replicating its card | 1 chip | Yes |

## TPU VM setup

Use standard CPython 3.13 (the repository default is 3.13.14) on a TPU VM. The
training recipes here target a v6e-4 and evaluation serves on one chip;
`model.mesh.shape` must multiply out to the number of visible chips. Do not use
the free-threaded `3.13t` build.

`scripts/setup_tpu_vm.sh` covers step 2 above. To do the same by hand:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
uv sync --frozen --extra test

python -c "import jax; print(jax.devices())"
```

Set `HF_TOKEN` before launch (Tunix's Hugging Face downloader expects a logged-in
session, including for public repositories). Tunix and its TPU JAX dependency
are installed by `pyproject.toml`; no CUDA packages, PyTorch trainer,
Accelerate, DeepSpeed, or GPU vLLM are used by the SFT stage.

Validate the environment on the TPU VM itself before starting a training job:

```bash
python -m open_r1_tpu.sft.preflight --config "$RECIPE"
```

This initializes JAX and requires the configured mesh device count to consist
entirely of TPUs. It also checks the installed Tunix/Optax APIs, loads the real
tokenizer, verifies the assistant-only chat-template boundary, and confirms the
recipe's merged export is supported for its model family.

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
./scripts/copy_gcs_bucket_data.sh --model Qwen2.5-1.5B --dataset SimpleRL-Zoo-Data
```

It reads `models/MODEL` and `datasets/NAME` from the bucket, writing them to
`models/MODEL` and `data/NAME` locally. `MODEL` comes from `--model` or
`$GCS_MODEL` and defaults to `Qwen2.5-Math-1.5B-RoPE-300k`; `NAME` comes from
`--dataset` or `$GCS_DATASET` and defaults to `OpenR1-Math-220k`. Set
`$GCS_MODEL_PREFIX`, `$GCS_DATA_PREFIX`, or `$GCS_DATA_GLOB` for a different
layout. Afterwards it reports how many Parquet shards the *training glob*
matches — not merely how many were copied — along with on-disk sizes, warning
rather than failing silently if either copy looks empty. `gcloud storage rsync`
is incremental, so re-running after an interrupted copy resumes cheaply.

Orbax checkpoints may use a `gs://` directory directly. Merged export is a
local-filesystem operation; export locally first, then copy the completed
directory to GCS with `gcloud storage rsync --recursive`.

### Continue text with a base model

A base model is a pretrained causal language model rather than a post-trained
chat model. To inspect its native next-token behavior, pass raw text directly
to `complete_qwen_tpu.py`. The script adds no system prompt, role markers, chat
template, or conversation history and generates at most 100 new tokens by
default:

```bash
source .venv/bin/activate
python scripts/complete_qwen_tpu.py \
  --model-path models/Qwen2.5-Math-1.5B \
  "The capital of France is"
```

Omit the quoted prompt for an interactive loop of independent completions. The
first completion compiles the TPU decode path and is slower than later ones.
Override the generation length with `--max-new-tokens`.

Generation is greedy and stops at the model's `<|endoftext|>` token or the
configured token limit. `--max-prompt-length` defaults to 2048. The completion
client deliberately uses ordinary masked attention: the pinned Tunix sampler
left-pads fixed-length prompts but does not pass the segment IDs that Qwen's
splash-attention path needs to exclude padding. This is slower than splash
attention, but prevents pad/EOS embeddings from changing the continuation.

### Chat with a model or a training run's own weights

`chat_qwen_tpu.py` applies the tokenizer's chat template. Point it at any
merged export:

```bash
python scripts/chat_qwen_tpu.py \
  --model-path artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged
```

To inspect a run before it has finished, pass its recipe instead, and the
latest checkpoint is restored on top of the base weights. A full fine-tune's
checkpoint replaces every parameter; a LoRA recipe's restores its adapters,
with the rank, alpha and target modules read from the recipe, because adapters
restored under a different geometry produce confident nonsense rather than an
error:

```bash
python scripts/chat_qwen_tpu.py \
  --recipe recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
```

Add `--step` to pick an earlier checkpoint, or `--checkpoint-dir` if the
artifacts moved. The restored step is printed on load, and it is rarely the
step the run stopped on: checkpoints are written every
`training.checkpointing_options.save_interval_steps` and only `max_to_keep` of
them survive. Asking for a step that was never written names the ones that
were.

**One TPU VM, one process.** The chat and completion clients use every visible
TPU chip in a topology derived from the model's `config.json` (Qwen2.5-Math-1.5B
gets an `[fsdp, tp] = [2, 2]` mesh on four chips), so do not run either beside
a training job: all chips are held by whichever process claims them first. If
an exported config has no `_name_or_path`, supply its canonical Tunix model
name with `--model-name`.

Two details of the chat script exist to match what training teaches, and both
would otherwise be silent:

- **The reply stops at `<|im_end|>`.** The ChatML template ends every turn with
  that token, but Qwen base models name `<|endoftext|>` as their EOS, and the
  sampler stops at the tokenizer's EOS unless told otherwise. Left alone the
  model runs past the end of its reply and writes your next turn for you.
- **An empty `<think></think>` block is hidden.** Qwen3's template opens an
  assistant turn with `<think>\n\n</think>\n\n` whenever the message carries
  no reasoning trace, so a corpus without traces teaches the model to emit that
  scaffold before every answer. It is stripped for display; a trace with actual
  content is left alone.

The system prompt defaults to empty. Pass `--system-prompt` with the text a
model was trained with (for the distillation recipe,
`recipes/Qwen2.5-Math-1.5B/system_prompt.txt`) to chat with it as it was
trained.

## Smoke test

Start with a short run before allocating a full training job. Every
`run_sft_tpu.sh` example below assumes `RECIPE` names the recipe, as in the
quick start:

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
  training.wandb.enabled=false \
  export.enabled=false
```

## Watching how training is going

Two signals, one quantitative and one qualitative.

**Held-out loss.** `dataset.eval_fraction` holds out a slice of the corpus, and
`dataset.eval_max_examples` caps it. The cap matters: a hundredth of a large
corpus is thousands of examples, and the trainer walks the whole eval set at
every evaluation, so an uncapped split would spend longer evaluating than
training. Evaluation runs every `training.eval_every_n_steps` and reports
`eval/loss` alongside `train/loss`. An evaluation also runs *before* the first
training step: the trainer takes a baseline reading whenever a held-out split
exists.

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

Each interval writes one JSON object per prompt to the recipe's
`training.transcripts.output_path`, recording the step, the prompt, the
completion, and flags for whether the reasoning trace was closed and whether
the token budget was exhausted. That last flag matters when reading the output:
a completion that used its whole budget was probably cut off, so a missing
`</think>` there is inconclusive rather than a real failure. The same records
go to W&B as a table under `samples/transcripts` unless
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

This summarizes whether the model is learning to close its reasoning traces:

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

`closed` should climb toward the prompt count over the first few thousand steps.
If it stays at zero while `train/loss` falls, that is the failure teacher-forced
loss cannot show, and it is worth stopping for.

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

Each recipe names its run (`training.run_name`) and group
(`training.wandb.group`). These values can be overridden normally:

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

## Long runs

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
export RECIPE=recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
mkdir -p artifacts
./scripts/run_sft_tpu.sh training.project_name="${WANDB_PROJECT}" \
  2>&1 | tee -a artifacts/train.log
```

Detach with `Ctrl-b d` and reattach with `tmux attach -t sft`. To stop a run,
interrupt the Python process itself (`Ctrl-c` in the pane, or kill its PID):
killing only the tmux session can leave the process holding the TPU.

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
mv artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/checkpoints \
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

This lives in
[`src/open_r1_tpu/core/logging.py`](src/open_r1_tpu/core/logging.py).
Quietening another package that hides behind absl is one entry in
`NOISY_PACKAGES`; a library with a logger of its own needs no help, since its
level can simply be set.

### Overriding the recipe

Any recipe value can be overridden using Tunix-style dotted arguments, for
example `training.max_steps=1000` or `dataset.max_length=16384`.

Important behavior:

- The input must contain a `messages` column whose final turn is an assistant
  response, as OpenR1-Math-220k's does. A corpus that names things
  differently — ShareGPT's `conversations` of `{from, value}` with
  `human`/`gpt` roles — is adapted by `dataset.messages_column` and
  `dataset.message_schema` in the recipe, not by code.
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
  incomplete, where dropping would exclude most of the data.
- `dataset.packing: true` packs whole examples into each window first-fit.
  Attention cannot cross example boundaries: per-token `segment_ids` gate the
  splash kernel, the non-flash path receives a block-diagonal causal mask, and
  RoPE positions restart per example.
- Increase `dataset.max_length` only after measuring peak HBM, and check how
  many complete reasoning traces survive length filtering.

## Reasoning distillation: Qwen2.5-Math-1.5B on OpenR1-Math-220k

[`recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml`](recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml)
trains the base DeepSeek distilled to make DeepSeek-R1-Distill-Qwen-1.5B, on
the public `open-r1/OpenR1-Math-220k` corpus, as open-r1 did at 7B. The quick
start runs it. The recipe header explains each choice; in short:

- **The whole corpus fits.** A 26,624-token window keeps every one of the
  93,554 examples that carry a complete `<think>` trace; a 4096 window would
  drop about 60%. `dataset.packing` fills each window with whole examples.
- **RoPE θ 300k.** The base declares a 4096-token context at θ 10000, and a
  26k window trained at that θ ended in repetition loops. The config-only
  RoPE-300k variant (`scripts/set_rope.py`, as open-r1 used at 7B), with a
  higher learning rate, took MATH-500 from 48% to 83%.
- **A full fine-tune.** The model ties its embeddings, so the rows for
  `<|im_end|>` live in the matrix LoRA freezes.
- **Four chips as `[fsdp, tp] = [2, 2]`.** Tensor parallelism halves the
  float32 logits of a 26k window, which do not fit one chip whole. Peak HBM was
  19.69 GiB per chip; a step takes about 10.2 s.

Its export scores 83.3% ± 1.1 on MATH-500 at tier 1 (three seeds, 16,384
tokens), against 79.8% for DeepSeek-R1-Distill-Qwen-1.5B under the same
protocol, and 83.7% under DeepSeek's card protocol against their published
83.9%.

## Benchmark evaluation

Held-out loss is teacher-forced: every scored token is conditioned on ground
truth, so it cannot say whether the model closes its reasoning trace, stops, or
reaches the right answer unaided. Those are the questions a reasoning stage is
judged on, and only free generation scored against a reference answers them.

The stack is three decoupled layers. Generation is vLLM on the TPU, serving the
merged export behind an OpenAI-compatible endpoint.
`src/open_r1_tpu/evaluation/experiment.py` sends it each document over the
`openai` SDK, `server.max_concurrency` requests at a time, then scores every
reply with LightEval's own metric objects as a library
(`src/open_r1_tpu/evaluation/scoring.py`), which for maths is symbolic
equivalence via latex2sympy2-extended rather than string equality. Given a
tracing config, it runs the same generation and scoring through Langfuse so
every document is traced; see
[Tracing in Langfuse](#tracing-in-langfuse-optional). Prompt templates, dataset coordinates, and metric configuration come
from a frozen task pack derived once from LightEval's own task registry
(`src/open_r1_tpu/evaluation/taskpack.py`, committed as `configs/taskpack.yaml`)
and re-verified at preflight, so a LightEval upgrade that moves one of them
fails loudly there instead of silently moving a headline number.
`src/open_r1_tpu/evaluation/run.py` and `reduce.py` own the reduction half —
they validate the recipe and reduce what the run wrote into a single summary —
and import neither JAX, Tunix, nor vLLM.

### Installing and running

Install the host harness and build the inference service in one re-runnable
step:

```bash
./scripts/setup_tpu_vm.sh --with-eval
```

This uses the tested uv 0.12.5 installer and
`uv sync --frozen --extra eval --extra test`, so the Python version, direct
evaluation dependencies, and complete transitive environment come from
`.python-version`, `pyproject.toml`, and the committed `uv.lock`. It then builds
the local vLLM TPU 0.27.0 service image from the digest-pinned Python base and
the hash-pinned lock in `docker/vllm-tpu/`. A missing or mismatched host
dependency is an evaluation preflight error rather than a warning recorded
after an expensive run.

**vLLM is not installed on the host.** Nothing in this package imports it: the
launcher starts the pinned container and everything after that is HTTP. The
container wrapper supplies the TPU requirements (`--privileged`, host
networking, and shared memory), bind-mounts the selected merged export read-only,
and persists Hugging Face and vLLM/XLA caches in the
`open-r1-tpu-huggingface-cache` and `open-r1-tpu-vllm-cache` Docker volumes.
It forwards `HF_TOKEN` by environment-variable name only when present; token
values never enter recipes, printed commands, or summaries.

The recipes omit `server.image`, which selects the local tag derived from the
committed `docker/vllm-tpu/Dockerfile` and `docker/vllm-tpu/vllm-tpu.lock`:

```yaml
server:
  serve_command: ["scripts/run_vllm_tpu_container.sh"]
```

Changing either build input changes the tag, so a stale local image cannot be
silently reused. The lock's hashes pin package contents; a local image ID is
recorded for inspection but is not a cross-machine identity. Build it explicitly
before a preflight or evaluation — the runner never surprises a benchmark with a
multi-minute build:

```bash
scripts/run_vllm_tpu_container.sh --build
scripts/run_vllm_tpu_container.sh --check
```

`docker/vllm-tpu/README.md` documents regenerating the lock. Do that only in
`python:3.12-slim-bookworm`, because `libtpu` has a `manylinux_2_31` wheel and
host-side resolution can select an unusable platform.

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
its transitive packages are no longer governed by this repository's lock.

Evaluation runs *after* training rather than beside it. Only one process can
hold the TPU chip, so stop the training job before starting the server.

### Preflight, then the smoke tier

```bash
python -m open_r1_tpu.evaluation.preflight \
  --config recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml
```

This checks the exact host dependency versions, Docker access, that the derived
local image exists, and the service versions installed in it, plus the export it
is pointed at and the recipe's task names. If the image is absent, preflight
names `scripts/run_vllm_tpu_container.sh --build`; it never builds implicitly.
Failures are caught before vLLM claims the TPU. A merged
export missing its tokenizer files or its chat template loads far enough to
serve requests and then answers off-distribution, producing a number that
measures the wrong thing. And Qwen base models name `<|endoftext|>` as their
EOS while the chat template closes turns with `<|im_end|>`, so a server left to the
tokenizer's own EOS runs past the end of every reply and writes the user's next
turn as well — which under a benchmark reads as a model that cannot stop
reasoning. A stop string cannot fix this: vLLM matches stop strings against
decoded text with special tokens stripped, so `<|im_end|>` as a stop string can
never fire on the real token. Preflight instead checks the setting that
actually governs it — the export's `generation_config.json` must name
`<|im_end|>`'s token id as an `eos_token_id` — and fails rather than warns when
it does not, since every benchmark number from such an export would be
invalid. Third, LightEval moves tasks between suites and releases, so a recipe naming one
that no longer exists is worth hearing about now rather than after the server has
spent fifteen minutes loading weights. A name that exists in a different suite is
reported with the suite it actually lives in.

```bash
RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

`scripts/run_eval_tpu.sh` owns the evaluation-level vLLM lifecycle: it builds
the container command from the recipe, waits for the server to answer, runs the
evaluation, and terminates its process group on the way out. The container
wrapper owns the Docker-level stop/removal. Both halves read the same recipe
and dotted overrides, so the image, port, model name, and context window cannot
drift apart. Evaluation summaries record the host stack, derived spec tag,
local image ID, digest-pinned base image, service package versions, and complete
constructed server command. Set `SKIP_SERVER=1` to reuse a server that is
already up.

### Tracing in Langfuse (optional)

Set `TRACE_CONFIG` to trace an evaluation in a self-hosted Langfuse:

```bash
TRACE_CONFIG=configs/tracing.yaml \
  RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

The run first syncs each of the recipe's tasks into a Langfuse dataset, then
runs one Langfuse experiment per task and seed, so every document's prompt,
completion and scores can be inspected and runs compared side by side. It
writes the same results files and summary as a local run. A dataset is named
by its task and a fingerprint of how it is asked and judged, plus `[:N]` when
the recipe caps it at `eval.max_samples`. `docker/langfuse/README.md` sets up
the stack and writes `configs/tracing.yaml`.

To update the evaluation environment intentionally, change the exact versions
in `pyproject.toml` and `src/open_r1_tpu/evaluation/stack.py`, run `uv lock`,
then repeat the unit suite and TPU smoke evaluation. Updating vLLM requires a
new hash-pinned service lock, which produces a new derived local tag, plus a
real TPU smoke run; never replace the digest-pinned Python base with `latest`.

### vLLM versus Tunix generation speed

An evaluation's seconds per sample do not establish which inference engine is
faster: it keeps `server.max_concurrency` requests in flight for vLLM's
continuous batching, while Tunix compiles a static batch shape and samples it
directly in-process. Run the controlled comparison on the TPU instead:

```bash
./scripts/benchmark_generation_tpu.sh
```

The launcher serves the export the eval recipe names (`server.model_path`, or
`MODEL_PATH`) with vLLM, measures it, releases the TPU, then loads the same
export into Tunix's direct `Sampler` and measures that. The default workload
covers batch/concurrency 1 and 8,
using the same 16 distinct, pre-rendered prompts, two measured repetitions,
greedy decoding, and 128 forced output tokens. One warm-up batch per shape is
excluded from steady-state throughput; server/model startup and warm-up times
are retained separately. Tunix flash attention is disabled for this short-prompt
inference workload, avoiding its 1024-token splash block padding requirement.

Raw results land in `generation-speed/{vllm,tunix}.json` beside the export.
`comparison.json` and `comparison.md` report output tokens per second, samples
per second, and the Tunix/vLLM ratio at each matching batch size. This is a speed
test only: EOS is deliberately ignored so both engines execute exactly the same
number of decode steps. Keep the ordinary evaluation path for termination and
accuracy.

### The tiers

`recipes/Qwen2.5-Math-1.5B/eval/` holds this project's tiers:

| Recipe | Cost | When |
| --- | --- | --- |
| `eval/tier0_smoke.yaml` | ~5 min | Every run. GSM8K, 200 problems, greedy. |
| `eval/tier1_core.yaml` | ~1 h | Every checkpoint worth keeping. MATH-500 over three seeds. |
| `eval/tier2_headline.yaml` | hours | Milestones. AIME24 and AIME25 over ten seeds. |
| `eval/tier3_regression.yaml` | hours | Milestones. IFEval and GPQA-Diamond over three seeds. |

The reference model's directory carries two more, `eval/tier4_code.yaml` and
`eval/tier5_gpqa.yaml`, and splits the tasks differently; see below.

Tier 0 does not measure ability; it catches a model that is broken in a way loss
cannot show. Tier 1 is the tier that decides whether a recipe change helped.
Tier 2 exists because AIME is the number the field quotes, not because 30
problems can settle an argument — one extra correct answer there moves pass@1 by
3.3 points. Tier 3 answers what the math-only corpus cost, which is an open
question for this project: `OpenR1-Math-220k` carries no instruction-following
or chat data at all.

Every tier `extends: base.yaml`, which holds what is genuinely identical across
them: the whole `server:` block except `max_model_len` — including
`server.turn_end_token`, the token the preflight verifies the export's
`generation_config.json` stops on, which differs per model family — and the
reporting markers and shared W&B settings. A tier file overrides only what actually
differs — its tasks, seeds, context window, output directory, and its whole
`sampling:` block, which stays in every tier file even where two tiers happen
to agree, because a measurement-affecting setting belongs where it is easy to
see. `open_r1_tpu.core.config.load_config` resolves `extends` (one level only)
before dotted overrides and validation run, so a base recipe is invisible to
both.

### The reference model replicates a published table

`recipes/DeepSeek-R1-Distill-Qwen-1.5B/eval/` is a different kind of recipe set
from the one above. It exists to reproduce the released model's own published
numbers on this stack, so it matches DeepSeek's protocol rather than this
project's tier vocabulary: no system prompt, its own turn-end token,
`reasoning_start: null`, and — on every tier carrying a published row at its
published protocol — their 32768-token generation budget and their sample
counts. The exception is tier 1, which instead adopts the project's tier-1
generation parameters so both models are measured on MATH-500 under one
protocol; its number is therefore not the card's 83.9.

| Published row | Value | Tier | Replicates |
| --- | --- | --- | --- |
| AIME 2024 pass@1 | 28.9 | `tier2_headline.yaml` | 64 |
| AIME 2024 cons@64 | 52.7 | `tier2_headline.yaml` | 64 |
| MATH-500 pass@1 | 83.9 | `tier1_core.yaml`\* | 3 |
| GPQA Diamond pass@1 | 33.8 | `tier5_gpqa.yaml` | 4 |
| LiveCodeBench pass@1 | 16.9 | `tier4_code.yaml` | 16 |
| CodeForces rating | 954 | — | — |

\*Tier 1 measures MATH-500 at the comparison protocol (16384-token budget,
three seeds), not the card's; the recipe header carries the overrides that
reproduce the published row.

CodeForces has no tier because it cannot be measured here: LightEval 0.13.0
ships no CodeForces task and no Elo harness, and the published rating is a
percentile placement against human contestants across ten Div.2 contests, not
a benchmark accuracy. Nothing in that directory approximates it.

Tiers 0 and 1 line up task-for-task with `recipes/Qwen2.5-Math-1.5B/eval/` —
tier 1 shares that tier's full generation-parameter set, which is what makes
the MATH-500 comparison direct. Tiers 2, 4 and 5 replicate the card instead,
which is why they do not line up with the project tiers of the same number.

To read the distillation export against the reference model's card, run the
reference directory's tiers with `server.model_path`,
`server.turn_end_token`, `server.extra_args` and `reporting.reasoning_start`
overridden to `recipes/Qwen2.5-Math-1.5B/eval/base.yaml`'s values.

**Tier 4 executes model-generated Python on the VM.** Scoring a code benchmark
means running the extracted solutions against the problem's tests, which
LightEval's `codegen_metrics` does in subprocesses behind its reliability
guard. That is the only way LiveCodeBench can be scored, and it is a real
difference in kind from every other tier, which only ever does arithmetic on
strings.

### Seeds are not optional

Seed variance alone moves small reasoning benchmarks by 5–15 points
([arXiv 2504.07086](https://arxiv.org/abs/2504.07086)), which is more than most
recipe changes are worth. Every task runs once per seed and is reported as mean
and standard deviation; `aggregate_across_seeds` reports a null standard
deviation at one seed rather than a reassuring `0.0`. Three seeds is the
documented minimum at MATH-500's size, ten at AIME's.

Replicates buy precision, not a different quantity. pass@1 is the probability
that a *single* sample is correct, and each replicate is an independent draw of
it, so the binomial standard error is `sqrt(p(1-p)/(N*n))` for `N` problems and
`n` replicates and shrinks as `1/sqrt(n)`. That is why the count is worth
choosing per benchmark rather than globally: at the reference model's published
scores, one replicate is ±8.3 points on AIME 2024's thirty problems but only
±1.6 on MATH-500's five hundred.

### cons@n needs the replicates together

A consensus number is the one metric that cannot be produced per replicate and
averaged: it is a majority vote *between* the replicates of a single problem,
and only then a judgement of the winner. `eval.consensus` asks for one, per
task, naming both the vote width and the metric that judges the winning answer:

```yaml
eval:
  seeds: [0, 1, ..., 63]
  consensus:
    "aime24|0":
      n: 64
      metric: "pass@k:k=1"
```

`open_r1_tpu.evaluation.consensus` performs the join at reduction time, over
the same per-document JSONL every other number is reduced from, so a
killed-and-resumed tier reduces to the same value. The vote is over answers
extracted by LightEval's own `math_normalizer`, not over raw completions —
voting on raw long-CoT text gives every sample its own group and quietly
degenerates into pass@1. A sample with no extractable answer does not vote;
`n` may not exceed the replicate count, which is checked at load time rather
than after the generations have been paid for.

`eval.seeds` indexes replicates; it does not seed them. The TPU backend refuses
a per-request seed outright — `TpuPlatform.validate_request` raises "JAX does
not support per-request seed." for any request vLLM classifies as
`RANDOM_SEED`, which is any seeded request with `temperature > 0` — and the
refusal arrives as an empty-body HTTP 500 that LightEval retries rather than
reports. So no seed is sent, replicates differ by the server's own RNG stream,
and the summary records `seeded_replicates: false`. Spread across replicates
remains a valid estimate of sampling variance, which is what the seeds are for;
reproducing an individual replicate is not available on this backend.

Published numbers are not a baseline either — they were produced by a different
stack. Measure the base model on this one:

```bash
RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier1_core.yaml ./scripts/run_eval_tpu.sh \
  server.model_path=models/Qwen2.5-Math-1.5B
```

Preflight will insist that the base's `generation_config.json` lists
`<|im_end|>`'s id as an EOS, as the export's does; add it to the base's copy
first.

### What the summary records

The run writes `summary_<tier>.json` under `reporting.output_dir`, holding the
per-task metrics aggregated across seeds, the sampling parameters, the model
path, and the installed versions of LightEval, openai and
latex2sympy2-extended. A result that does not name its stack cannot be compared
with one produced months later.
`reporting.summary_path` accepts a `gs://` URI to land it beside the checkpoint
it scored. Any consensus number lands under `summary["consensus"]`, apart from
`tasks_metrics`: it is a single value computed from every replicate jointly and
has no spread across them, so filing it beside the seed-aggregated metrics
would invite reading its absent standard deviation as one-replicate noise.
`summary_rows` flattens both, so W&B receives it either way.

Alongside accuracy it records four things read out of the runner's own
per-document records (`src/open_r1_tpu/evaluation/reduce.py`), each diagnosing
a failure accuracy alone cannot separate:

- `truncation_rate` — the fraction whose completion ended with
  `finish_reason == "length"`, a fact the server states rather than a
  `token_count >= max_new_tokens` inference. A truncated trace scores as wrong
  and reads as a reasoning failure, so if this is not near zero then
  `sampling.max_new_tokens` is too low and the accuracy under it is not
  trustworthy.
- `reasoning_closed_rate` and `answer_marker_rate` — whether the model produces
  the shape SFT was teaching, independent of whether the answer is right. A
  recipe sets `reporting.reasoning_start: null` when the model's chat template
  opens the reasoning block inside the prompt itself (DeepSeek's distills append
  `<think>` to the generation prompt), so a completion can only ever carry the
  closing tag and closure is judged on that alone.
- `mean_completion_tokens` — the length-inflation signal, from the API
  response's own `usage`.

`mean_completion_tokens` (and, with it, `truncation_rate`) is `null` only if
every document in a seed's run is missing a `completion_tokens` field, which
the API response always carries -- unlike the detail-Parquet field this project
used to depend on for it.

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

Training writes resumable Tunix/Orbax checkpoints to
`training.checkpoint_dir` (for the distillation recipe,
`artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/checkpoints`). At successful
completion it writes a standard safetensors model to `export.output_dir`
(`artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged`): the live parameters of
a full fine-tune, or a LoRA run's adapters merged into the base weights.

The default `export.overwrite=true` replaces that specific merged-output
directory on a repeated run; checkpoint and model-cache directories are guarded
against accidental use as export targets.

Use that merged directory as the starting point for GRPO: set a GRPO recipe's
`model.model_path` to it. GRPO loads it twice, as the policy (with a fresh
LoRA adapter) and as the frozen reference whose KL term keeps policy updates
close to the distilled model rather than the original base.

A full fine-tune's export maps Tunix parameter names back to Hugging Face ones
for Qwen2 and Qwen3 (`open_r1_tpu.model.export`). Merged LoRA export uses
Tunix's model-specific exporter for Qwen3. The pinned
Tunix has none for Qwen2, so `open_r1_tpu.model.export` supplies one on
Tunix's generic merge (`save_qwen2_lora_merged_model_as_safetensors`, which
reuses Qwen3's key rules because the two families name and lay out their
projections alike); `tests/test_qwen2_lora_export.py` checks it against a
staged Qwen2.5-1.5B on the VM. For other families, set `export.enabled=false`
unless that Tunix params module provides
`save_lora_merged_model_as_safetensors`.

## GRPO

`python -m open_r1_tpu.grpo.run` trains a LoRA policy with Tunix's GRPO
learner. `src/open_r1_tpu/grpo/run.py` loads the policy and reference and
builds the RL cluster, `data.py` loads prompts with their gold answers, and
`rewards.py` holds the reward functions a recipe selects with
`grpo.reward_functions`. The tested recipe reproduces SimpleRL-Zoo on
Qwen2.5-1.5B on a v6e-4:

```bash
hf download hkust-nlp/SimpleRL-Zoo-Data --repo-type dataset \
  --include "simplelr_abel_level3to5/*" --local-dir data/SimpleRL-Zoo-Data
hf download Qwen/Qwen2.5-1.5B --local-dir models/Qwen2.5-1.5B
python -m open_r1_tpu.grpo.run \
  --config recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml
```

Its 400 steps take about 14 hours and end with a merged export in
`artifacts/Qwen2.5-1.5B-SimpleRL-Zoo/grpo/merged`. Qwen2 GRPO on the pinned
Tunix needs `model.dtype: float32` and `model.use_flash_attention: false`:
splash attention and bfloat16 both corrupt the rollouts, as the recipe header
explains. If a multi-chip start fails with `START_SESSION failed` after an
earlier run crashed, `LIBTPU_INIT_ARGS=--noenable_tpunetd_client` lets libtpu
build the slice itself.

`recipes/Qwen2.5-Math-1.5B/grpo/dapo-math-17k.yaml` runs GRPO on
a distilled SFT export with `open-r1/DAPO-Math-17k-Processed` prompts; it has
not been run.

### A positive control: reproducing SimpleRL-Zoo on Qwen2.5-1.5B

`recipes/Qwen2.5-1.5B/` checks the GRPO pipeline against a
published RL result. SimpleRL-Zoo (arXiv 2503.18892) trained the Qwen2.5-1.5B
base with GRPO and a correctness-only reward on 8,523 MATH level 3-5 problems
and reports GSM8K 55.7 → 74.4 and MATH-500 29.6 → 59.0; their trained model
is public. Scored the same way, this recipe's export reaches GSM8K 68.2 (first
200 problems) and MATH-500 54.7, against 74.5 and 57.2 for their checkpoint.

- `grpo/simplerl-zoo.yaml` copies their data, plain-text "Abel" prompt (as a
  chat template), stop tokens, 1/0 reward (`math_answer_reward`, which reads
  the answer in their order) and GRPO settings (8 rollouts, temperature 1.0,
  KL 1e-4 with the low-variance estimator, token-mean loss). It cannot copy
  their scale: 16 prompts per step on a v6e-4 against their 1,024, LoRA
  against a full fine-tune, and 2,048 new tokens against 8,192. The header
  lists every difference.
- `eval/tier1_core.yaml` scores GSM8K and MATH-500 at their generation
  settings (temperature 1.0, top-p 0.95, 16,000 tokens), over three seeds.
  Run it on the base, their checkpoint and this run's export, each staged
  with `scripts/stage_eval_model.py`. That copies a model directory with the
  GRPO recipe's chat template and stop tokens, so vLLM serves every model
  the prompt it was trained on; `eval/base.yaml` has the commands.

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
