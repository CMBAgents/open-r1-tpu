# open-r1-tpu

TPU-native supervised reasoning distillation built with JAX, Grain, Orbax, and
[Google Tunix](https://github.com/google/tunix). The pipeline mirrors the SFT
stage in Hugging Face's `open-r1`: it loads conversational reasoning traces,
applies the model's chat template, and trains a causal language model to emit
the complete assistant reasoning trace before a later GRPO stage.

The default recipe trains `Qwen/Qwen3-1.7B-Base` with LoRA on
`open-r1/Mixture-of-Thoughts` using a v5e-8 TPU. It supervises assistant tokens
only, filters incomplete or overlength reasoning traces, writes resumable Orbax
checkpoints, and exports a merged Hugging Face-style safetensors directory.

## TPU VM setup

Use Python 3.11 or 3.12 on a TPU VM with eight visible devices:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'

python - <<'PY'
import jax
print(jax.devices())
assert len(jax.devices()) == 8
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

This initializes JAX and requires the recipe's eight devices to be TPUs. It also
checks the installed Tunix/Optax APIs, loads the real Qwen tokenizer, verifies
the assistant-only chat-template boundary, and confirms that Tunix supplies the
Qwen3 merged-LoRA exporter.

## Smoke test

Start with a short run before allocating a full training job:

```bash
./scripts/run_sft_tpu.sh \
  dataset.max_examples=128 \
  dataset.max_length=2048 \
  training.max_steps=4 \
  training.gradient_accumulation_steps=1 \
  training.checkpointing_options.save_interval_steps=2
```

The first step includes JAX/XLA compilation and will be much slower than later
steps.

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
  dataset.max_length=4096 \
  model.mesh.shape='[1, 8]' \
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
- `dataset.max_length=8192` is a conservative starting point. The Open-R1 CUDA
  recipe uses 32768, but that should be increased only after measuring HBM use
  on the chosen TPU topology.

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
