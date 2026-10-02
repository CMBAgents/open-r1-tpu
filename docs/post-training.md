# What is post-training?

A [base model](base-models.md) can only continue text. *Post-training* is
everything done after pretraining to make it useful: following the chat
format, stopping when it has answered, and, for a reasoning model, working
through a problem before it commits to an answer. It uses a tiny fraction of
pretraining's compute: the SFT run here takes about 19 hours on four TPU
chips.

## From a base model to a reasoning model

DeepSeek trained [DeepSeek-R1](https://arxiv.org/abs/2501.12948) in stages.
Reinforcement learning applied straight to the base model (DeepSeek-R1-Zero)
taught it to reason, but its reasoning was hard to read. R1 itself began with
a little supervised fine-tuning on readable examples, then alternated
reinforcement learning with further fine-tuning. Finally, DeepSeek fine-tuned
small open models on 800,000 of R1's answers; these *distilled* models
reasoned better than the same small models trained with reinforcement
learning alone.

[Open-R1](https://github.com/huggingface/open-r1) rebuilds those stages in
the open, and this pipeline runs them on TPUs:

1. **Supervised fine-tuning (SFT)** on reasoning written by a stronger model
   teaches the format and the habit of reasoning.
2. **Reinforcement learning (RL)** on problems with checkable answers
   improves the model on its own attempts.
3. **[Benchmarking](benchmarking.md)** measures the result on problems it has
   never seen.

## Supervised fine-tuning (SFT)

### The idea

SFT continues the model's next-token training, but on example conversations
that show the behaviour you want. The model learns to imitate them. It is
never told whether its own answers are right; it only learns to make the
examples more likely.

When the examples are a stronger model's reasoning, this is *distillation*.
The [SFT recipe](sft.md) trains Qwen2.5-Math-1.5B on
[OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k):
maths problems with solutions written by DeepSeek-R1, of which 93,554 have a
complete `<think> ... </think>` trace. Each training example is a
conversation whose last turn is the reply to learn:

```text
<|im_start|>system
You are an assistant that solves problems by reasoning carefully ...<|im_end|>
<|im_start|>user
Find all real x such that ...<|im_end|>
<|im_start|>assistant
<think>
Let me start by ... (often thousands of tokens)
</think>
The solutions are ... <|im_end|>
```

### What happens in a training step

1. **Tokenise.** The chat template turns each conversation into tokens.
2. **Mask.** Only the reply's tokens, including its closing `<|im_end|>`,
   count towards the loss. The system prompt and question are context: the
   model should learn to answer questions, not to write them. This is
   *assistant-only loss*. The reply's start is found from the exact output of
   the chat template, never guessed; an example where it cannot be found is
   dropped.
3. **Pack.** Short examples are packed together into fixed-length windows so
   little of each window is padding. Each example keeps its own positions,
   and attention never crosses from one example into another.
4. **Score.** The loss is how surprised the model is by each reply token, on
   average: the negative log-probability it gave the right token.
5. **Update.** The gradient says how each weight should change to lower the
   loss, and the optimiser (AdamW) changes it a little in that direction.

Examples longer than the window are dropped rather than cut, since a cut
trace would teach the model to stop mid-thought. A held-out slice of the data
is scored every few hundred steps (`eval/loss`); if it rose while the
training loss kept falling, the model would be memorising its examples rather
than learning from them.

### Full fine-tuning and LoRA

A *full fine-tune* updates every weight. *LoRA* (low-rank adaptation)
freezes the weights and trains a small pair of matrices beside each one,
a few per cent as many numbers, which saves memory; the pair is merged back
into the weights at export. The SFT recipe is a full fine-tune because
Qwen2.5-Math-1.5B uses one matrix for both its input and output embeddings,
and the `<|im_end|>` row it must learn sits in the matrix LoRA would freeze.

### What SFT can and cannot teach

Imitation teaches a format very well: to reason inside `<think>` tags, to
box the answer, and to stop. But the
model can at best match its examples, and it never learns from its own
mistakes. That is what reinforcement learning adds.

[Run SFT →](sft.md)

## Reinforcement learning (RL)

### The idea

In reinforcement learning the model writes its own answers, a program checks
them, and the model is nudged towards the answers that scored well. It needs
no worked solutions, only questions and a way to check an answer. Maths
suits it, because a final answer can be checked automatically; this is
*reinforcement learning with verifiable rewards*.

### GRPO step by step

GRPO (Group Relative Policy Optimisation) is the method DeepSeek used for R1.
Each step:

1. **Sample.** For each question, the model writes a *group* of answers
   (eight in the recipes here), at a temperature so that they differ.
2. **Score.** Reward functions give each answer a number.
3. **Compare.** Each answer's *advantage* is how much better it scored than
   the rest of its group: its reward minus the group's mean, divided by the
   group's standard deviation.
4. **Update.** Answers with a positive advantage are made more likely, and
   those with a negative one less likely.

Say a group of eight answers has three right (reward 1) and five wrong
(reward 0). The mean is 0.375 and the standard deviation 0.52, so each
right answer gets an advantage of +1.21 and each wrong one −0.72. The group is
its own baseline, so GRPO needs no second network to predict how good an
answer should have been, which makes it simpler than older methods such as
PPO.

If every answer in a group is right, or every one wrong, every advantage is
zero and the group teaches nothing. Questions far too easy or far too hard
waste the step; training logs the fraction of groups with mixed rewards as
`signal/groups_mixed_frac`.

### Rewards

The reward is the only thing that tells the model what you want, and the
model will exploit any gap in it. `src/open_r1_tpu/grpo/rewards.py` has four,
which a recipe picks with `grpo.reward_functions`:

| Function | Scores |
| --- | --- |
| `math_answer_reward` | 1 if the answer equals the right one, else 0. SimpleRL-Zoo's reward, used by the [RL recipe](grpo.md). |
| `answer_correctness_reward` | 1 if the final number equals the right one, else 0 |
| `correctness_reward` | Full marks for a right `\boxed{}` answer, partial credit for a close one, a penalty for a wrong one |
| `format_reward` | Full marks for a closed `<think>` block followed by a `\boxed{}` answer, with partial credit for each part |

### The policy and the reference

GRPO holds two copies of the starting model:

- the **policy**, the model being trained. Only a LoRA adapter trains, and it
  is merged into the weights at export.
- the **reference**, a frozen copy. A *KL penalty* (`grpo.beta`) measures how
  far the policy's token probabilities have moved from the reference's and
  counts against the reward, so the model cannot forget everything else in
  pursuit of it.

A *clip* (`grpo.epsilon`) also limits how far one update can change any
answer's probability.

### RL with and without SFT first

DeepSeek-R1-Zero showed that RL alone can teach a base model to reason, and
[SimpleRL-Zoo](https://arxiv.org/abs/2503.18892) repeated that on small open
models. The [RL recipe](grpo.md) reproduces SimpleRL-Zoo from the
Qwen2.5-1.5B base. RL can equally start from an SFT model, as the
[tutorials](tutorials.md) do, so that it refines a model that already has the
format.

[Run RL →](grpo.md)

## SFT and RL compared

| | SFT | RL (GRPO) |
| --- | --- | --- |
| Needs | Worked examples | Questions and a way to check an answer |
| Learns from | Someone else's answers | Its own answers, and how they scored |
| Teaches | Format and style; it can match its examples | Accuracy; it can go beyond any examples |
| Cost of a step | One forward and backward pass | Generating a group of answers per question first, which dominates |
| Weights trained here | All of them | A LoRA adapter |

## How training runs on a TPU

Both stages run the same way:

1. **The recipe is loaded and checked.** A misspelt key fails at once rather
   than hours into a run; see [Recipes](recipes.md).
2. **Tunix builds the model** on the recipe's mesh of TPU chips, from the
   local base model.
3. **JAX compiles the training step** with XLA for that layout. This makes
   the first step much slower than the rest.
4. **Training runs**, logging losses (or rewards) to Weights & Biases and
   writing Orbax checkpoints. A restarted run resumes from the newest
   checkpoint by itself.
5. **The model is exported** in Hugging Face's format, ready for
   [benchmarking](benchmarking.md) or [chat](chat.md).

[Training in depth](training.md) covers each of these.
