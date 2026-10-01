# open-r1-tpu

Reasoning post-training for open language models on Google Cloud TPUs, with
JAX and [Google Tunix](https://github.com/google/tunix): supervised
distillation (SFT), GRPO, and benchmark evaluation served by vLLM and scored
with LightEval's metrics. It follows the SFT-to-GRPO workflow of
[Open-R1](https://github.com/huggingface/open-r1) without CUDA, PyTorch or TRL.

Measured on TPU v6e:

- **SFT.** Distilling Qwen2.5-Math-1.5B on OpenR1-Math-220k scores 83.3% on
  MATH-500 (three seeds, 16k-token budget), against 79.8% for
  DeepSeek-R1-Distill-Qwen-1.5B measured the same way.
- **GRPO.** Reproducing SimpleRL-Zoo on Qwen2.5-1.5B, with LoRA and a
  sixty-fourth of their batch, reaches GSM8K 68.2 (first 200 problems) and
  MATH-500 54.7, against 74.5 and 57.2 for their published checkpoint scored
  the same way.
- **Evaluation.** Scored on this stack, SimpleRL-Zoo's checkpoint lands within
  0.1 (GSM8K) and 1.8 (MATH-500) points of its published numbers.

## What is here

| Path | Contents |
| --- | --- |
| `src/open_r1_tpu/sft/` | SFT data preparation, training and preflight |
| `src/open_r1_tpu/grpo/` | GRPO prompt loading, rewards and training |
| `src/open_r1_tpu/evaluation/` | Evaluation: recipe config, vLLM server, generation, scoring, summary, optional Langfuse tracing |
| `src/open_r1_tpu/model/` | Model and tokenizer loading, LoRA, optimizer, metrics, export, checkpoint restore |
| `src/open_r1_tpu/core/` | Recipe loading, the shared command line, logging |
| `recipes/` | Training and evaluation recipes; see [Recipes](#recipes) |
| `scripts/` | Launchers and tools: VM setup, SFT, evaluation, the vLLM container, chat, export |
| `docker/` | The pinned vLLM TPU image, and the optional Langfuse stack |
| `configs/` | The frozen LightEval task pack, and the tracing config template |
| `docs/` | [Training](docs/training.md), [evaluation](docs/evaluation.md), and [chat](docs/chat.md) guides |

Run every command from the repository root on the TPU VM. Only one process can
hold a TPU, so training, evaluation and chat run one at a time.

## Install

```bash
git clone https://github.com/CMBAgents/open-r1-tpu.git
cd open-r1-tpu
./scripts/setup_tpu_vm.sh            # add --with-eval for the evaluation stack
```

The script installs `uv`, the CPython pinned in `.python-version`, and the
locked environment in `.venv`, then checks JAX sees the TPU and runs the unit
tests. `--with-eval` also installs LightEval and builds the vLLM image;
`--skip-verify` skips the checks when another job holds the TPU. It is safe to
re-run.

It also writes `~/.open-r1-tpu.env`, kept out of the repository, with the
settings to fill in commented out. Set the ones you need and load it with the
environment:

```bash
export HF_TOKEN=hf_...                  # Hub downloads
export WANDB_ENTITY=your-user-or-team   # W&B
export WANDB_PROJECT=your-project
```

```bash
source ~/.open-r1-tpu.env
source .venv/bin/activate
wandb login
```

The shipped recipes log to Weights & Biases. To run without it, pass
`training.wandb.enabled=false` to training or `reporting.wandb.enabled=false`
to an evaluation.

## Quick start: distillation

[`recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml`](recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml)
fine-tunes Qwen2.5-Math-1.5B, the base DeepSeek distilled, on
OpenR1-Math-220k. It needs a v6e-4. Its header explains each setting; in
short:

- **A 26,624-token window** keeps all 93,554 examples with a complete
  `<think>` trace, packed whole into each window. A 4,096 window would drop
  about 60%.
- **RoPE θ 300k.** At the base's θ of 10,000, a 26k window ended in repetition
  loops. Raising θ in `config.json` (as Open-R1 did at 7B), with a higher
  learning rate, took MATH-500 from 48% to 83%.
- **A full fine-tune.** The model ties its embeddings, so the `<|im_end|>` rows
  it must learn sit in the matrix LoRA would freeze.
- **`[fsdp, tp] = [2, 2]` over four chips.** Tensor parallelism halves the
  float32 logits of a 26k window, which do not fit on one chip. Peak HBM is
  19.7 GiB per chip.

Stage the data and the base, then make the long-context variant:

```bash
hf download open-r1/OpenR1-Math-220k --repo-type dataset \
  --include 'data/*' --local-dir data/OpenR1-Math-220k
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B-RoPE-300k
python scripts/set_rope.py models/Qwen2.5-Math-1.5B-RoPE-300k \
  --rope-theta 300000 --max-position-embeddings 32768
```

Check the environment, run a four-step smoke test, then train:

```bash
export RECIPE=recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
python -m open_r1_tpu.sft.preflight --config "$RECIPE"

python -m open_r1_tpu.sft.run --config "$RECIPE" \
  dataset.max_examples=128 training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2 \
  training.checkpoint_dir=/tmp/sft-smoke/checkpoints \
  training.metrics_log_dir=/tmp/sft-smoke/logs \
  training.wandb.enabled=false export.enabled=false

python -m open_r1_tpu.sft.run --config "$RECIPE" training.project_name="${WANDB_PROJECT}"
```

The full run is 6,710 steps, about 19 hours, and ends with a merged export in
`artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged`. The smoke test writes its
checkpoints to `/tmp` because training resumes from the newest checkpoint in
`training.checkpoint_dir`. The first step includes XLA compilation and is much
slower than the rest. Run long jobs under `tmux`, and stop one by interrupting
its Python process: killing only the tmux session can leave it holding the
TPU. [docs/training.md](docs/training.md) covers monitoring, transcripts,
W&B, resuming, overrides and export.

## GRPO

GRPO trains a LoRA adapter with Tunix's GRPO learner and keeps the starting
model as the frozen KL reference. `grpo/run.py` builds the policy, reference
and RL cluster, `grpo/data.py` loads prompts with their gold answers, and
`grpo/rewards.py` holds the reward functions a recipe picks with
`grpo.reward_functions`.

[`recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml`](recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml)
reproduces SimpleRL-Zoo (arXiv 2503.18892) on Qwen2.5-1.5B: their data,
prompt, stop tokens, 1/0 correctness reward and GRPO settings, at the scale a
v6e-4 allows. Its comments note each difference at the setting concerned. 400 steps take about 14 hours
and end with a merged export in `artifacts/Qwen2.5-1.5B-SimpleRL-Zoo/grpo/merged`.

```bash
hf download hkust-nlp/SimpleRL-Zoo-Data --repo-type dataset \
  --include "simplelr_abel_level3to5/*" --local-dir data/SimpleRL-Zoo-Data
hf download Qwen/Qwen2.5-1.5B --local-dir models/Qwen2.5-1.5B
python -m open_r1_tpu.grpo.run --config recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml
```

`recipes/Qwen2.5-1.5B/eval/tier1_core.yaml` scores GSM8K and MATH-500 at
SimpleRL-Zoo's settings. Stage each model it compares (the base, their
checkpoint, this export) with `scripts/stage_eval_model.py`, which gives it
the GRPO recipe's chat template and stop tokens; `eval/base.yaml` has the
commands and how the reported numbers were served.

On the pinned Tunix, Qwen2 GRPO needs `model.dtype: float32` and
`model.use_flash_attention: false`: bfloat16 and splash attention both corrupt
the rollouts. To start from an SFT export, set the recipe's `model.model_path`
to it, and `model.rope_theta` to the SFT recipe's value: the loader reads the
recipe, not the export's `config.json`. If a multi-chip start fails with `START_SESSION failed` after an
earlier run crashed, `LIBTPU_INIT_ARGS=--noenable_tpunetd_client` lets libtpu
build the slice itself.

## Evaluation

Evaluation serves the merged export with vLLM in a pinned container, sends
each benchmark problem to it, and scores the replies with LightEval's metrics.
It needs the `--with-eval` environment:

```bash
python -m open_r1_tpu.evaluation.preflight \
  --config recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml
RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

`run_eval_tpu.sh` starts the server, waits for it, evaluates and stops it.
Results land in the recipe's `reporting.output_dir`: one JSONL file per seed
and task, and a `summary_<tier>.json` with each metric's mean and standard
deviation across seeds, truncation and format rates, and any cons@n. Pass
dotted overrides after the command, such as `server.model_path=...` to score
another model; [docs/evaluation.md](docs/evaluation.md#run) shows how to
prepare a base model for that.

| Tier | Cost | Contents |
| --- | --- | --- |
| `tier0_smoke.yaml` | ~5 min | GSM8K, 200 problems, greedy: catches a broken model |
| `tier1_core.yaml` | ~1 h | MATH-500 over three seeds: decides whether a change helped |
| `tier2_headline.yaml` | hours | AIME24 and AIME25 over ten seeds |
| `tier3_regression.yaml` | hours | IFEval and GPQA-Diamond: what the maths corpus cost |

Set `TRACE_CONFIG=configs/tracing.yaml` to trace every document in a
self-hosted Langfuse ([docker/langfuse/README.md](docker/langfuse/README.md));
the results are the same either way. [docs/evaluation.md](docs/evaluation.md)
covers the protocol: preflight checks, seeds, cons@n, the reference model's
card replication and the summary fields.

## Recipes

Recipes live in `recipes/<base model>/<stage>/<dataset>.yaml`, named after the
Hugging Face model the run starts from. Evaluation tiers are
`recipes/<model>/eval/tier<N>_<name>.yaml` and extend that directory's
`base.yaml`. Each recipe's header says how to stage and run it. Outputs go
under `artifacts/<output model name>/`.

| Recipe | What it does | Hardware | Run |
| --- | --- | --- | --- |
| `Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml` | Reasoning distillation, full fine-tune | v6e-4 | Yes |
| `Qwen2.5-Math-1.5B/eval/` | Tiers 0-3 for that export | 1 chip | Tiers 0-1 |
| `Qwen2.5-1.5B/grpo/simplerl-zoo.yaml` | GRPO positive control against SimpleRL-Zoo | v6e-4 | Yes |
| `Qwen2.5-1.5B/eval/` | SimpleRL-Zoo's tier-1 protocol | 1 chip | Yes |
| `DeepSeek-R1-Distill-Qwen-1.5B/eval/` | The reference model, replicating its card | 1 chip | Tiers 0-2 and 5 |

Any value can be overridden on the command line with a dotted argument, such
as `training.max_steps=1000`. Recipes reject unknown keys, so a typo fails at
load time; only `model`, `tokenizer` and `training.checkpointing_options` pass
through to Tunix unchecked. Deployment-specific values (buckets, W&B entities, hosts) belong in
the environment or in overrides, never in a recipe.

## Chat with a model

```bash
python scripts/chat_tpu.py --model-path artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged \
  --model-name qwen2.5-math-1.5b
python scripts/chat_tpu.py --recipe recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
```

The first form loads any merged export, named with its Tunix model name; the
second restores an SFT run's latest checkpoint on top of its base, so a run
can be inspected before it finishes.
`scripts/complete_tpu.py` continues raw text without a chat template. See
[docs/chat.md](docs/chat.md).

## Development

```bash
uv sync --frozen --extra test --extra eval --extra dev
pre-commit install
python -m pytest
pre-commit run --all-files
```

The unit tests need no TPU; tests that need JAX, Tunix or the `eval` extra
skip without them. Two groups are deselected by default:
`python -m pytest -m integration` needs a live vLLM server, and
`python -m pytest -m network` downloads from the Hugging Face Hub. Ruff and
pyright run as pre-commit hooks, and CI (`.github/workflows/ci.yml`) runs the hooks and the
tests on every push and pull request. `uv sync` also installs console scripts
for the entry points: `open-r1-tpu-sft`, `open-r1-tpu-sft-preflight`,
`open-r1-tpu-grpo`, `open-r1-tpu-eval`, `open-r1-tpu-eval-preflight` and
`open-r1-tpu-taskpack`. [AGENTS.md](AGENTS.md) holds the rules for changing
the code.

Every dependency is pinned in `uv.lock`, and the evaluation stack is pinned
exactly (`src/open_r1_tpu/evaluation/stack.py`): change the pins, the lock and
`docker/vllm-tpu` together.
