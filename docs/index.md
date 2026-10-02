# open-r1-tpu

Reasoning post-training for open language models on Google Cloud TPUs, with
JAX and [Google Tunix](https://github.com/google/tunix): supervised
distillation (SFT), GRPO, and benchmark evaluation served by vLLM and scored
with LightEval's metrics. It follows the SFT-to-GRPO workflow of
[Open-R1](https://github.com/huggingface/open-r1) without CUDA, PyTorch or TRL.

## Results

Measured on TPU v6e:

| Stage | What was run | Result | Compared with |
| --- | --- | --- | --- |
| SFT | Qwen2.5-Math-1.5B distilled on OpenR1-Math-220k | MATH-500 83.3% (three seeds, 16k-token budget) | DeepSeek-R1-Distill-Qwen-1.5B: 79.8%, measured the same way |
| GRPO | SimpleRL-Zoo reproduced on Qwen2.5-1.5B, with LoRA and a sixty-fourth of their batch | GSM8K 68.2 (first 200 problems), MATH-500 54.7 | Their published checkpoint, scored the same way: 74.5 and 57.2 |
| Evaluation | SimpleRL-Zoo's checkpoint scored on this stack | Within 0.1 (GSM8K) and 1.8 (MATH-500) points of its published numbers | |

## Where to start

<div class="grid cards" markdown>

-   **New to fine-tuning?**

    ---

    Two notebooks train a small model with SFT and then GRPO on one TPU chip,
    explaining each step.

    [Tutorials →](tutorials.md)

-   **Set up a TPU VM**

    ---

    Clone the repository, install the locked environment and check JAX sees
    the TPU.

    [Install →](getting-started.md)

-   **Distil a reasoning model**

    ---

    Fine-tune Qwen2.5-Math-1.5B on OpenR1-Math-220k on a v6e-4, from staging
    the data to a merged export.

    [Distillation →](sft.md)

-   **Train with GRPO**

    ---

    Reproduce SimpleRL-Zoo on Qwen2.5-1.5B with a LoRA adapter and a
    correctness reward.

    [GRPO →](grpo.md)

-   **Score a model**

    ---

    Serve an export with vLLM in a pinned container and score it with
    LightEval's metrics, from a five-minute smoke test to AIME.

    [Evaluation →](evaluation.md)

-   **Write a recipe**

    ---

    How recipes are laid out, validated and overridden from the command line.

    [Recipes →](recipes.md)

</div>

## How it fits together

A run starts from a recipe: one YAML file per base model, stage and dataset.
Training (SFT or GRPO) writes resumable Orbax checkpoints and, when it
finishes, a merged Hugging Face export. Evaluation serves that export with
vLLM on the TPU and scores its replies; chat loads it, or a checkpoint from a
run still in progress, for a conversation.

Only one process can hold a TPU, so training, evaluation and chat run one at
a time. Run every command from the repository root on the TPU VM.
