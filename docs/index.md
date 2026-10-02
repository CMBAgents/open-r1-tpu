# open-r1-tpu: a post-training pipeline on TPU

open-r1-tpu takes an open language model that can only continue text and
trains it to be an assistant that reasons before it answers. It runs on
Google Cloud TPUs, with JAX and [Google Tunix](https://github.com/google/tunix).

- **Based on [Open-R1](https://github.com/huggingface/open-r1)** by Hugging
  Face, whose workflow, datasets and evaluation it follows. We train a model
  to be an assistant, and to reason.
- **Supported by** the Google TPU Builder Program.
- **Built by** [Rowan d'Auria](https://www.linkedin.com/in/rowan-d-auria).

## What is Open-R1?

In January 2025 DeepSeek released
[DeepSeek-R1](https://arxiv.org/abs/2501.12948), a model that works through a
problem step by step before it answers, and that performed comparably to
OpenAI's o1 on maths and coding benchmarks. They published the weights and
described the training, but not the data or the code that produced it.

[Open-R1](https://github.com/huggingface/open-r1) is Hugging Face's
open-source replication of that training pipeline. It rebuilds each stage in
the open: datasets of R1's reasoning, supervised fine-tuning that distils it
into smaller models, reinforcement learning with GRPO, and the benchmarks to
check the result.

Open-R1 is written for NVIDIA GPUs: it uses CUDA, PyTorch and Hugging Face's
TRL. open-r1-tpu keeps the same workflow and replaces that stack, so it runs
on TPUs:

| | Open-R1 | open-r1-tpu |
| --- | --- | --- |
| Hardware | NVIDIA GPUs (CUDA) | Google Cloud TPUs |
| Framework | PyTorch | JAX |
| Training library | TRL | Tunix |
| Benchmark serving | vLLM on GPU | vLLM on TPU, in a pinned container |
| Benchmark scoring | LightEval | LightEval's metrics, used as a library |

### Results so far

Measured on TPU v6e:

| Stage | What was run | Result | Compared with |
| --- | --- | --- | --- |
| SFT | Qwen2.5-Math-1.5B distilled on OpenR1-Math-220k | MATH-500 83.3% (three seeds, 16k-token budget) | DeepSeek-R1-Distill-Qwen-1.5B: 79.8%, measured the same way |
| RL | SimpleRL-Zoo reproduced on Qwen2.5-1.5B, with LoRA and a sixty-fourth of their batch | GSM8K 68.2 (first 200 problems), MATH-500 54.7 | Their published checkpoint, scored the same way: 74.5 and 57.2 |
| Benchmarking | SimpleRL-Zoo's checkpoint scored on this stack | Within 0.1 (GSM8K) and 1.8 (MATH-500) points of its published numbers | |

<figure markdown="span">
  ![MATH-500 accuracy of the SFT checkpoints rises from 70.9% at step 1,000 to 83.3% at step 6,710, passing DeepSeek-R1-Distill-Qwen-1.5B's 79.8%, while the share of replies cut off at the token limit falls to 7.4%.](assets/figures/sft-math500-by-step-light.svg#only-light)
  ![MATH-500 accuracy of the SFT checkpoints rises from 70.9% at step 1,000 to 83.3% at step 6,710, passing DeepSeek-R1-Distill-Qwen-1.5B's 79.8%, while the share of replies cut off at the token limit falls to 7.4%.](assets/figures/sft-math500-by-step-dark.svg#only-dark)
</figure>

[Results](results.md) has every figure, with the numbers behind it.

## Setting up on TPU

The pipeline runs on Google Cloud. Three pieces are needed:

- **Buckets.** A Cloud Storage bucket holds models, datasets, checkpoints and
  finished models, so they outlive any one VM.
- **VMs.** Training runs on a TPU VM. The full recipes use a v6e-4: one host
  with four TPU v6e chips.
- **A small evaluation VM.** Benchmarking serves a model on a single chip, so
  a one-chip v6e-1 can score models while the v6e-4 keeps training.

[Setting up on TPU](tpu-setup.md) walks through creating each one, and
[Install](install.md) through installing the software on the VM.

## Getting started

!!! tip "New to fine-tuning or reinforcement learning?"

    Two [tutorial notebooks](tutorials.md) train a small model with SFT and
    then RL on one TPU chip, explaining each step.

### What is a base model?

A base model is a language model straight out of *pretraining*: it has read
trillions of words from the internet, books and code, and learnt to predict
the next word. It knows a great deal, but it can only continue text. Ask it a
question and it may answer, write more questions, or carry on indefinitely.
Every run here starts from an open base model such as
[Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B).
[Base models](base-models.md) explains what is inside one.

### Downloading a base model

Base models are downloaded from the [Hugging Face Hub](https://huggingface.co)
with its `hf` command:

```bash
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B
```

[Base models](base-models.md#downloading-a-base-model) covers access tokens,
copying a model from a bucket, and preparing it for long reasoning traces.

### What is post-training?

Post-training is everything after pretraining that turns a base model into a
useful one. It is far cheaper than pretraining, and it is where a model learns
to follow the chat format, to stop when it has answered, and to reason. This
pipeline has two stages, in the order DeepSeek and Open-R1 use them:

- **Supervised fine-tuning (SFT).** The model is shown example conversations
  and learns to imitate them. Trained on reasoning written by a stronger
  model, it learns to reason the same way; this is called *distillation*.
  [How SFT works →](post-training.md#supervised-fine-tuning-sft)
- **Reinforcement learning (RL).** The model writes its own answers, a program
  checks them, and the model is nudged towards the ones that scored well. It
  needs only questions and a way to check an answer. This pipeline uses GRPO,
  the method behind DeepSeek-R1.
  [How RL works →](post-training.md#reinforcement-learning-rl)

[What is post-training?](post-training.md) explains both in more detail, and
the [SFT](sft.md) and [RL](grpo.md) guides run them.

### Benchmarking

A falling training loss does not show that a model can solve problems on its
own. Benchmarking does: the model answers a fixed set of problems with known
answers, such as school maths (GSM8K), competition maths (MATH-500, AIME) or
science questions (GPQA), and its answers are scored. Here vLLM serves the
model on the TPU and LightEval's metrics score each reply, over several seeds
so that luck is not mistaken for progress. [Benchmarking](benchmarking.md)
explains how, and [Running evaluations](evaluation.md) runs one.

#### What is Langfuse?

[Langfuse](https://langfuse.com) is an open-source platform for recording
what a language model was asked, what it answered and how each answer was
scored. Connected to an evaluation, it shows every prompt, reply and score in
a web interface, so you can read why a model got a problem wrong and compare
runs side by side. It is optional and self-hosted; see
[Langfuse](langfuse.md).

## What's next

Frontier training now teaches models to use tools: this is *agentic* AI. An
agentic model does not answer from memory alone. It takes actions, such as
running code, searching or calling another program, and fetches the
resources it needs as it works. It is trained with the same reinforcement
learning used here, except that each attempt becomes a sequence of actions
and their results, and the reward judges whether the task was done. Training
models to use tools on TPUs is the next step for this pipeline.
