# GRPO

GRPO trains a LoRA adapter with Tunix's GRPO learner and keeps the starting
model as the frozen KL reference.

| Module | Role |
| --- | --- |
| `grpo/run.py` | Builds the policy, reference and RL cluster |
| `grpo/data.py` | Loads prompts with their gold answers |
| `grpo/rewards.py` | The reward functions a recipe picks with `grpo.reward_functions` |

## Reproducing SimpleRL-Zoo

[`recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml`](https://github.com/CMBAgents/open-r1-tpu/blob/main/recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml)
reproduces SimpleRL-Zoo ([arXiv 2503.18892](https://arxiv.org/abs/2503.18892))
on Qwen2.5-1.5B: their data, prompt, stop tokens, 1/0 correctness reward and
GRPO settings, at the scale a v6e-4 allows. Its comments note each difference
at the setting concerned.

```bash
hf download hkust-nlp/SimpleRL-Zoo-Data --repo-type dataset \
  --include "simplelr_abel_level3to5/*" --local-dir data/SimpleRL-Zoo-Data
hf download Qwen/Qwen2.5-1.5B --local-dir models/Qwen2.5-1.5B
python -m open_r1_tpu.grpo.run --config recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml
```

400 steps take about 14 hours and end with a merged export in
`artifacts/Qwen2.5-1.5B-SimpleRL-Zoo/grpo/merged`. It reaches GSM8K 68.2
(first 200 problems) and MATH-500 54.7, against 74.5 and 57.2 for their
published checkpoint scored the same way, with a sixty-fourth of their batch.

## Scoring it

`recipes/Qwen2.5-1.5B/eval/tier1_core.yaml` scores GSM8K and MATH-500 at
SimpleRL-Zoo's settings. Stage each model it compares (the base, their
checkpoint, this export) with `scripts/stage_eval_model.py`, which gives it
the GRPO recipe's chat template and stop tokens; `eval/base.yaml` has the
commands and how the reported numbers were served. See
[Evaluation](evaluation.md) for running a tier.

## Constraints on the pinned Tunix

!!! warning "Qwen2 needs float32 and no flash attention"

    On the pinned Tunix, Qwen2 GRPO needs `model.dtype: float32` and
    `model.use_flash_attention: false`: bfloat16 and splash attention both
    corrupt the rollouts.

**Starting from an SFT export.** Set the recipe's `model.model_path` to the
export, and `model.rope_theta` to the SFT recipe's value: the loader reads the
recipe, not the export's `config.json`. GRPO loads the export twice: as the
policy, with a new LoRA adapter, and as the frozen reference whose KL term
keeps the policy close to it.

**`START_SESSION failed`.** If a multi-chip start fails with this after an
earlier run crashed, `LIBTPU_INIT_ARGS=--noenable_tpunetd_client` lets libtpu
build the slice itself.
