# Evaluation

Held-out loss is teacher-forced, so it cannot say whether a model closes its
reasoning, stops, or reaches the right answer on its own. Evaluation answers
that by generating freely and scoring against a reference.

It has three layers:

- **Serving.** vLLM runs on the TPU in a pinned container and serves the
  merged export over an OpenAI-compatible API. `evaluation/server.py` builds
  its command from the recipe.
- **Generation.** `evaluation/run.py` sends each problem to the server with
  the `openai` SDK, `server.max_concurrency` requests at a time
  (`evaluation/generate.py`).
- **Scoring.** Each reply is scored with LightEval's metric objects, used as a
  library (`evaluation/scoring.py`); maths answers are compared symbolically,
  not as strings. `evaluation/summary.py` reduces the per-problem records to a
  summary.

Prompts, dataset revisions and metric settings come from a task pack frozen
from LightEval's registry (`configs/taskpack.yaml`, written by
`evaluation/taskpack.py`). Preflight re-checks it against the installed
LightEval, so an upgrade that changes a task fails there rather than moving a
number. Nothing on the host imports JAX, Tunix or vLLM.

## Install

```bash
./scripts/setup_tpu_vm.sh --with-eval
```

This installs the locked `eval` extra and builds the vLLM TPU image from
`docker/vllm-tpu/`: a digest-pinned Python base and a hash-pinned lock. The
image tag is derived from the Dockerfile, the lock and the patches, so a
stale image is never reused. Rebuild or check it by hand with:

```bash
scripts/run_vllm_tpu_container.sh --build
scripts/run_vllm_tpu_container.sh --check
```

vLLM is not installed on the host. The container wrapper gives the container
the TPU (`--privileged`, host networking, shared memory), mounts the export
read-only, keeps Hugging Face and XLA caches in Docker volumes, and forwards
`HF_TOKEN` by name only. It uses `sudo docker` when the login user is not in
the Docker group, and stops the container on interrupt so the TPU is not left
held.

To use your own vLLM install instead, override the command and turn off the
image; preflight then warns that the environment is unchecked:

```yaml
server:
  serve_command: ["/opt/vllm-venv/bin/vllm", "serve"]
  image: null
```

To change the evaluation stack, update the pins in `pyproject.toml` and
`src/open_r1_tpu/evaluation/stack.py` together, run `uv lock`, and repeat the
unit tests and a TPU smoke evaluation. `docker/vllm-tpu/README.md` covers
regenerating the vLLM lock.

## Run

Stop any training job first: only one process can hold the TPU.

```bash
python -m open_r1_tpu.evaluation.preflight \
  --config recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml
RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

Preflight checks, before vLLM takes the TPU:

- the host package versions, Docker access, the image and the vLLM versions
  inside it (it never builds the image itself);
- the recipe's tasks against the task pack;
- the export: weights, tokenizer files, chat template, and that
  `generation_config.json` lists the turn-end token as an EOS.

The last check matters for Qwen: the base models' EOS is `<|endoftext|>`, but
the chat template ends turns with `<|im_end|>`. A server that stops only on
the former writes past the end of every reply, which reads as a model that
cannot stop reasoning. A stop string cannot fix it, because vLLM matches stop
strings against text with special tokens removed.

`run_eval_tpu.sh` starts the server, waits for it, evaluates, and stops it.
The server command and the evaluation read the same recipe and overrides, so
they cannot disagree on the model, port or context window. Pass overrides
after the command; set `SKIP_SERVER=1` to use a server that is already up.

To measure a base model on the same stack, download it and add `<|im_end|>`'s
id (151645 for Qwen2.5) to the `eos_token_id` list in its
`generation_config.json`, then run preflight and the tier against it:

```bash
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B
# edit models/Qwen2.5-Math-1.5B/generation_config.json
RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier1_core.yaml ./scripts/run_eval_tpu.sh \
  server.model_path=models/Qwen2.5-Math-1.5B
```

`run_eval_tpu.sh` does not run preflight itself, so an unedited base writes
past the end of every reply without any error.

## Tracing in Langfuse (optional)

```bash
TRACE_CONFIG=configs/tracing.yaml \
  RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

With a tracing config, each task is synced into a Langfuse dataset and each
task and seed runs as a Langfuse experiment, so every prompt, completion and
score can be inspected and runs compared. The results files and summary are
the same as a local run's. A dataset is named after its task and a
fingerprint of its prompt and metrics, plus `[:N]` when `eval.max_samples`
caps it. [docker/langfuse/README.md](../docker/langfuse/README.md) sets up the
stack and writes `configs/tracing.yaml` (template:
`configs/tracing.example.yaml`).

## Tiers

`recipes/Qwen2.5-Math-1.5B/eval/`:

| Recipe | Cost | Use |
| --- | --- | --- |
| `tier0_smoke.yaml` | ~5 min | Every run. GSM8K, 200 problems, greedy. Catches a broken model. |
| `tier1_core.yaml` | ~1 h | Every checkpoint worth keeping. MATH-500, three seeds. Decides whether a change helped. |
| `tier2_headline.yaml` | hours | Milestones. AIME24 and AIME25, ten seeds. The number the field quotes, though one problem is 3.3 points. |
| `tier3_regression.yaml` | hours | Milestones. IFEval and GPQA-Diamond, three seeds. What a maths-only corpus cost. |

Every tier extends `base.yaml`, which holds what the tiers share: the `server`
block (apart from `max_model_len`) and the reporting settings. Each tier sets
its own tasks, seeds, context window, output directory and full `sampling`
block, so settings that change the measurement are visible in the tier file.
`extends` goes one level deep and is resolved before overrides and
validation.

`recipes/Qwen2.5-1.5B/eval/tier1_core.yaml` scores GSM8K and MATH-500 at
SimpleRL-Zoo's settings.

## The reference model

`recipes/DeepSeek-R1-Distill-Qwen-1.5B/eval/` is set up to reproduce
DeepSeek-R1-Distill-Qwen-1.5B's published numbers on this stack. It runs
without a system prompt, as DeepSeek advise, and the tiers that match a
published row use DeepSeek's 32,768-token budget and sample counts.

| Published row | Value | Tier | Replicates |
| --- | --- | --- | --- |
| AIME 2024 pass@1 | 28.9 | `tier2_headline.yaml` | 64 |
| AIME 2024 cons@64 | 52.7 | `tier2_headline.yaml` | 64 |
| MATH-500 pass@1 | 83.9 | `tier1_core.yaml`\* | 3 |
| GPQA Diamond pass@1 | 33.8 | `tier5_gpqa.yaml` | 4 |
| LiveCodeBench pass@1 | 16.9 | `tier4_code.yaml` | 16 |
| CodeForces rating | 954 | none | none |

\*Tier 1 instead matches `recipes/Qwen2.5-Math-1.5B/eval/tier1_core.yaml`
(16,384 tokens, three seeds), so the two models compare directly; its header
gives the overrides for the card's protocol. Tiers 0 and 1 line up with the
Qwen2.5-Math-1.5B tiers; tiers 2, 4 and 5 follow the card instead.

CodeForces cannot be measured here: LightEval has no CodeForces task, and the
rating is a placement against human contestants, not an accuracy.

**Tier 4 runs model-written Python on the VM.** LiveCodeBench is scored by
executing the extracted solutions against the problem's tests, in
subprocesses behind LightEval's reliability guard. Tiers 0, 1, 2 and 5 have
been run; tier 4 has not.

## Seeds and consensus

Seed variance alone moves small reasoning benchmarks by 5-15 points
([arXiv 2504.07086](https://arxiv.org/abs/2504.07086)). Every task runs once
per seed and is reported as a mean and standard deviation; one seed reports a
`null` spread, not `0.0`. The pass@1 standard error is
`sqrt(p(1-p)/(N*n))` for `N` problems and `n` seeds, so choose the count per
benchmark: at the reference model's scores one seed is ±8.3 points on AIME
2024's 30 problems but ±1.6 on MATH-500's 500.

`eval.seeds` numbers replicates; it does not seed them. vLLM's TPU backend
rejects a per-request seed whenever temperature is above zero, so no seed is
sent, replicates differ by the server's own randomness, and the summary
records `seeded_replicates: false`. The spread is still a valid estimate of
sampling variance, but a single replicate cannot be reproduced.

A consensus score (cons@n) is a majority vote across one problem's
replicates, so it is computed from all seeds together rather than per seed:

```yaml
eval:
  seeds: [0, 1, 2, 3]  # in practice all 64, as in tier2_headline.yaml
  consensus:
    "aime24|0":
      n: 4
      metric: "pass@k:k=1"
```

The vote is over answers extracted by LightEval's `math_normalizer`, not raw
completions, since every long reasoning trace is unique. A completion with no
extractable answer does not vote. `n` cannot exceed the number of seeds; this
is checked when the recipe loads.

## The summary

The run writes one JSONL file per seed and task and `summary_<tier>.json`
under `reporting.output_dir`. The summary holds each task's metrics as a mean
and standard deviation across seeds, the sampling settings, the model path,
the vLLM image, and the installed version of every evaluation package.
Consensus scores sit under `consensus`, apart from the seed means, since each
is a single value. `reporting.summary_path` accepts a `gs://` URI.

Under `generation` (and per seed under `per_seed_generation`) it records six
diagnostics:

- `truncation_rate`: the fraction of completions the server cut off at
  `sampling.max_new_tokens`. A truncated trace scores as wrong, so if this is
  not near zero the budget is too low and the accuracy is not trustworthy.
- `format_rate`, `reasoning_closed_rate` and `answer_marker_rate`: whether
  the model produces the format SFT taught (a closed reasoning block and the
  answer marker; each alone), whether or not the answer is right. Set
  `reporting.reasoning_start: null` when the chat template opens the reasoning
  block in the prompt, as DeepSeek's distills do.
- `mean_completion_tokens` and `mean_completion_chars`: the length-inflation
  signal.

`truncation_rate` is `null` when the server returned no finish reasons, and
`mean_completion_tokens` when it returned no token counts; neither is
estimated from characters. The other rates are `null` only when no completion
succeeded.

Set `reporting.wandb.run_id` to the training run's W&B id to log these on the
same run as the loss curves. W&B resumes by id, not by name, so without one
they go to a new run.

## vLLM versus Tunix generation speed

An evaluation's time per sample does not compare the engines, because vLLM
batches many concurrent requests while Tunix samples a fixed batch. For a
controlled comparison:

```bash
./scripts/benchmark_generation_tpu.sh
```

It serves the eval recipe's export (`server.model_path`, or `MODEL_PATH`) with
vLLM and measures it, frees the TPU, then does the same with Tunix's
`Sampler`. Both see the same 16 prompts at batch sizes 1 and 8, greedy, with
exactly 128 output tokens (EOS ignored), after one warm-up batch per size.
Results go to `generation-speed/` beside the export, with `comparison.md`
giving tokens and samples per second and the Tunix/vLLM ratio. It measures
speed only.

## Limits

- **Only generative tasks.** An OpenAI-compatible server returns no
  log-probabilities for a given continuation, so LightEval's loglikelihood
  tasks (plain MMLU, HellaSwag, ARC) cannot run. Tier 3 is generative for
  that reason.
- **One chip per server.** The recipes serve with `tensor_parallel_size: 1`;
  they have run on a v6e-1 and on one chip of a v6e-4. On a multi-chip VM,
  pin the server with `TPU_VISIBLE_CHIPS=0 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
  TPU_PROCESS_BOUNDS=1,1,1`; the container wrapper passes every `TPU_*` and
  `LIBTPU_*` variable through.
