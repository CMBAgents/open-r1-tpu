# Recipes

A recipe is one YAML file that holds every setting of a training run or an
evaluation. Each recipe's header says how to stage and run it.

## Layout

```text
recipes/<base model>/<stage>/<dataset>.yaml     # training: sft or grpo
recipes/<model>/eval/tier<N>_<name>.yaml        # evaluation tiers
recipes/<model>/eval/base.yaml                  # what a model's tiers share
```

The directory is the Hugging Face name of the model the run starts from.
Evaluation tiers extend that directory's `base.yaml`. Outputs go under
`artifacts/<output model name>/`.

## Shipped recipes

| Recipe | What it does | Hardware | Run |
| --- | --- | --- | --- |
| [`Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml`](https://github.com/CMBAgents/open-r1-tpu/blob/main/recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml) | Reasoning distillation, full fine-tune ([guide](sft.md)) | v6e-4 | Yes |
| [`Qwen2.5-Math-1.5B/eval/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/recipes/Qwen2.5-Math-1.5B/eval) | Tiers 0-3 for that export ([tiers](evaluation.md#tiers)) | 1 chip | Tiers 0-1 |
| [`Qwen2.5-1.5B/grpo/simplerl-zoo.yaml`](https://github.com/CMBAgents/open-r1-tpu/blob/main/recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml) | GRPO positive control against SimpleRL-Zoo ([guide](grpo.md)) | v6e-4 | Yes |
| [`Qwen2.5-1.5B/eval/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/recipes/Qwen2.5-1.5B/eval) | SimpleRL-Zoo's tier-1 protocol | 1 chip | Yes |
| [`DeepSeek-R1-Distill-Qwen-1.5B/eval/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/recipes/DeepSeek-R1-Distill-Qwen-1.5B/eval) | The reference model, replicating its card ([details](evaluation.md#the-reference-model)) | 1 chip | Tiers 0-2 and 5 |

The tutorial notebooks have their own recipes in
[`examples/recipes/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/examples/recipes).

## Overrides

Any value can be overridden on the command line with a dotted argument:

```bash
python -m open_r1_tpu.sft.run --config "$RECIPE" \
  training.max_steps=1000 training.wandb.enabled=false
```

Overrides apply before validation. A recipe can also `extends:` one base
recipe, by a path relative to itself; `extends` goes one level deep and is
resolved before overrides.

## Validation

Recipes reject unknown keys and suggest the closest match, so a typo fails at
load time rather than hours into a run. The checked sections are `dataset`,
`optimizer`, `training` (with `training.wandb` and `training.transcripts`),
`export`, and for GRPO `grpo` and `rollout`. Only `model`, `tokenizer` and
`training.checkpointing_options` pass through to Tunix unchecked.

The product of `model.mesh.shape` must equal the number of visible JAX
devices, and `axis_names` must have the same rank.

!!! note "Keep recipes neutral"

    Deployment-specific values (buckets, W&B entities, hosts) belong in the
    environment or in overrides, never in a recipe.

## Writing one

- Raise `dataset.max_length` only after measuring peak HBM on the target TPU,
  and check how many examples survive the length filter.
- Keep a finite `training.max_steps`, and check `num_train_epochs` supplies
  enough examples after filtering.
- Orbax checkpoints may target GCS; merged export must target a local
  directory and be copied to GCS afterwards.

[Training in depth](training.md#recipes) covers the SFT data format, and
[Evaluation](evaluation.md#tiers) how tiers share a base.
