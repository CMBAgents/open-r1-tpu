# Distillation (SFT)

[`recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml`](https://github.com/CMBAgents/open-r1-tpu/blob/main/recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml)
fine-tunes Qwen2.5-Math-1.5B, the base DeepSeek distilled, on
OpenR1-Math-220k. It needs a v6e-4 and scores 83.3% on MATH-500.

## The recipe

Its header explains each setting; in short:

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

## Stage the data and the base

Download both, then make the long-context variant of the base:

```bash
hf download open-r1/OpenR1-Math-220k --repo-type dataset \
  --include 'data/*' --local-dir data/OpenR1-Math-220k
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B-RoPE-300k
python scripts/set_rope.py models/Qwen2.5-Math-1.5B-RoPE-300k \
  --rope-theta 300000 --max-position-embeddings 32768
```

If they are already in a GCS bucket, copy them instead; see
[Training in depth](training.md#data-from-a-gcs-bucket).

## Check, smoke test, train

```bash
export RECIPE=recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
python -m open_r1_tpu.sft.preflight --config "$RECIPE"
```

The preflight checks the TPU chip count, the installed Tunix and Optax APIs,
the tokenizer's assistant boundary and export support. Then run four steps:

```bash
python -m open_r1_tpu.sft.run --config "$RECIPE" \
  dataset.max_examples=128 training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2 \
  training.checkpoint_dir=/tmp/sft-smoke/checkpoints \
  training.metrics_log_dir=/tmp/sft-smoke/logs \
  training.wandb.enabled=false export.enabled=false
```

!!! warning "Keep smoke output out of `artifacts/`"

    Training resumes from the newest checkpoint in `training.checkpoint_dir`,
    so a full run pointed at the smoke run's directory would continue from its
    four steps. That is why the smoke test writes to `/tmp`.

The first step includes XLA compilation and is much slower than the rest.
Then train:

```bash
python -m open_r1_tpu.sft.run --config "$RECIPE" training.project_name="${WANDB_PROJECT}"
```

The full run is 6,710 steps, about 19 hours, and ends with a merged export in
`artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged`. Scored as it trained, it
passed DeepSeek's own distillation of the same base:

<figure markdown="span">
  ![MATH-500 accuracy of the SFT checkpoints rises from 70.9% at step 1,000 to 83.3% at step 6,710, passing DeepSeek-R1-Distill-Qwen-1.5B's 79.8%, while the share of replies cut off at the token limit falls to 7.4%.](assets/figures/sft-math500-by-step-light.svg#only-light)
  ![MATH-500 accuracy of the SFT checkpoints rises from 70.9% at step 1,000 to 83.3% at step 6,710, passing DeepSeek-R1-Distill-Qwen-1.5B's 79.8%, while the share of replies cut off at the token limit falls to 7.4%.](assets/figures/sft-math500-by-step-dark.svg#only-dark)
</figure>

[Results](results.md#supervised-fine-tuning) has the numbers, and how the
finished model compares with DeepSeek's published scores.

!!! danger "Stopping a run"

    Run long jobs under `tmux`, and stop one by interrupting its Python
    process (`Ctrl-c` in the pane, or kill its PID). Killing only the tmux
    session can leave it holding the TPU.

## Next steps

- [Training in depth](training.md): monitoring, transcripts, W&B, resuming,
  overrides and export.
- [Evaluation](evaluation.md): score the export.
- [Chat](chat.md): talk to it, or to a checkpoint before the run finishes.
- [GRPO](grpo.md): continue from the export with reinforcement learning.
