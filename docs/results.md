# Results

Every number on this page was measured on TPU v6e with this project's
evaluation stack, described in [Benchmarking](benchmarking.md). The numbers
behind each figure are in the table under it.

## Supervised fine-tuning

The [SFT recipe](sft.md) trains Qwen2.5-Math-1.5B on OpenR1-Math-220k. Its
checkpoints were scored on MATH-500 as training went on, alongside
DeepSeek-R1-Distill-Qwen-1.5B, DeepSeek's own distillation of the same base,
measured the same way.

<figure markdown="span">
  ![Two line charts against training step. Left: MATH-500 accuracy of the learning-rate 2e-4 run rises from 70.9% at step 1,000 to 83.3% at step 6,710, passing DeepSeek-R1-Distill-Qwen-1.5B's 79.8%; the learning-rate 4e-5 run levels off near 71%. Right: the share of replies cut off at the token limit falls from 21.5% to 7.4% for the 2e-4 run, against 4.5% for DeepSeek's model.](assets/figures/sft-math500-by-step-light.svg#only-light)
  ![Two line charts against training step. Left: MATH-500 accuracy of the learning-rate 2e-4 run rises from 70.9% at step 1,000 to 83.3% at step 6,710, passing DeepSeek-R1-Distill-Qwen-1.5B's 79.8%; the learning-rate 4e-5 run levels off near 71%. Right: the share of replies cut off at the token limit falls from 21.5% to 7.4% for the 2e-4 run, against 4.5% for DeepSeek's model.](assets/figures/sft-math500-by-step-dark.svg#only-dark)
</figure>

The higher learning rate is most of the difference: at 4e-5 the model
plateaued near 71%. The two panels move together. A reply cut off at the
16,384-token limit has no final answer and scores as wrong, and most cut-off
replies were the model repeating itself; as training taught the model to
finish its reasoning and stop, fewer replies were cut off and more were right.

??? note "The numbers"

    MATH-500 pass@1 and the share of replies cut off at the 16,384-token
    limit, in per cent: mean ± standard deviation over three runs at
    temperature 0.6.

    | Model | Step | MATH-500 | Cut off |
    | --- | ---: | ---: | ---: |
    | Learning rate 4e-5 | 500 | 63.7 ± 0.1 | 35.3 ± 1.9 |
    | Learning rate 4e-5 | 2,000 | 69.8 ± 1.2 | 27.5 ± 0.4 |
    | Learning rate 4e-5 | 5,000 | 71.2 ± 1.7 | 24.7 ± 1.3 |
    | Learning rate 2e-4 | 1,000 | 70.9 ± 1.6 | 21.5 ± 0.9 |
    | Learning rate 2e-4 | 2,000 | 78.3 ± 2.0 | 13.1 ± 1.4 |
    | Learning rate 2e-4 | 6,710 | **83.3 ± 1.1** | 7.4 ± 0.5 |
    | DeepSeek-R1-Distill-Qwen-1.5B | | 79.8 ± 1.3 | 4.5 ± 0.8 |

## Against DeepSeek's model card

DeepSeek published scores for DeepSeek-R1-Distill-Qwen-1.5B. Running their
model on this stack with their settings checks the stack; running ours the
same way compares the two models.

<figure markdown="span">
  ![Dot plot of accuracy on four benchmarks. MATH-500: our SFT model 83.7%, level with DeepSeek's published 83.9%, while DeepSeek's model measured here scores 78.6%. AIME 2024: ours 24.3%, DeepSeek's model 30.6%, published 28.9%. AIME 2024 majority of 64: ours 43.3%, DeepSeek's model 53.3%, published 52.7%. GPQA-Diamond: ours 22.7%, DeepSeek's model 28.3%, published 33.8%.](assets/figures/deepseek-card-light.svg#only-light)
  ![Dot plot of accuracy on four benchmarks. MATH-500: our SFT model 83.7%, level with DeepSeek's published 83.9%, while DeepSeek's model measured here scores 78.6%. AIME 2024: ours 24.3%, DeepSeek's model 30.6%, published 28.9%. AIME 2024 majority of 64: ours 43.3%, DeepSeek's model 53.3%, published 52.7%. GPQA-Diamond: ours 22.7%, DeepSeek's model 28.3%, published 33.8%.](assets/figures/deepseek-card-dark.svg#only-dark)
</figure>

On AIME, DeepSeek's model scores what its card says, within a couple of
points. On MATH-500 and GPQA-Diamond it scores about five points under its
card, so this stack does not reproduce every published number. Our SFT model
matches the card on MATH-500, the kind of problem it was trained on, and
falls behind on the harder AIME and on GPQA's science questions. On GPQA it
closed its reasoning block in only 27% of replies, against 98% for
DeepSeek's model: away from competition maths, it often gives an answer
without finishing its reasoning.

??? note "The numbers"

    Accuracy in per cent on DeepSeek's protocol: a 32,768-token budget,
    temperature 0.6 and no system prompt. Mean ± standard deviation over 4
    runs (MATH-500, GPQA-Diamond) or 64 runs (AIME 2024). The majority vote
    is one number over all 64 runs.

    | Benchmark | DeepSeek's card | DeepSeek's model, measured here | Our SFT model |
    | --- | ---: | ---: | ---: |
    | MATH-500 | 83.9 | 78.6 ± 0.3 | 83.7 ± 1.2 |
    | AIME 2024 | 28.9 | 30.6 ± 6.4 | 24.3 ± 5.3 |
    | AIME 2024, majority of 64 | 52.7 | 53.3 | 43.3 |
    | GPQA-Diamond | 33.8 | 28.3 ± 0.0 | 22.7 ± 0.4 |

## How much one run varies

AIME has only 30 problems, so one problem is worth 3.3 points, and a model
that samples its answers can score very differently from one run to the
next.

<figure markdown="span">
  ![Two dot plots of 64 runs each on AIME 2024. DeepSeek-R1-Distill-Qwen-1.5B's runs range from 16.7% to 40.0% around a mean of 30.6%; our SFT model's range from 13.3% to 33.3% around a mean of 24.3%.](assets/figures/aime-runs-light.svg#only-light)
  ![Two dot plots of 64 runs each on AIME 2024. DeepSeek-R1-Distill-Qwen-1.5B's runs range from 16.7% to 40.0% around a mean of 30.6%; our SFT model's range from 13.3% to 33.3% around a mean of 24.3%.](assets/figures/aime-runs-dark.svg#only-dark)
</figure>

Each model's runs span 20 points or more. A single run could put our model
ahead of DeepSeek's, or more than ten points below its mean; 64 runs pin each
mean down to within about a point. This is why every benchmark here runs
several times; see [Benchmarking](benchmarking.md#how-answers-are-scored).

??? note "The numbers"

    AIME 2024 pass@1 in per cent, over 64 runs at temperature 0.6 with a
    32,768-token budget.

    | Model | Mean | Standard deviation | Lowest run | Highest run |
    | --- | ---: | ---: | ---: | ---: |
    | DeepSeek-R1-Distill-Qwen-1.5B | 30.6 | 6.4 | 16.7 | 40.0 |
    | Our SFT model | 24.3 | 5.3 | 13.3 | 33.3 |

## Reinforcement learning

The [RL recipe](grpo.md) reproduces SimpleRL-Zoo: GRPO with a 1/0
correctness reward on the Qwen2.5-1.5B base, here with a LoRA adapter and a
sixty-fourth of their batch. It scored a held-out set every 50 steps while it
trained.

<figure markdown="span">
  ![Line chart over 350 GRPO steps. The share of held-out answers that are correct rises from 5.9% at step 0 to 45.7% by step 100 and 51.8% at step 350. The share with a boxed final answer rises from 29% to 96% by step 100 and stays near 99%.](assets/figures/grpo-heldout-by-step-light.svg#only-light)
  ![Line chart over 350 GRPO steps. The share of held-out answers that are correct rises from 5.9% at step 0 to 45.7% by step 100 and 51.8% at step 350. The share with a boxed final answer rises from 29% to 96% by step 100 and stays near 99%.](assets/figures/grpo-heldout-by-step-dark.svg#only-dark)
</figure>

Most of the gain comes in the first 100 steps. Part of it is format: the
grader reads the `\boxed{}` answer first, and boxed answers went from 29% to
96%. But the number of problems solved at least once in eight tries also
rose, from 19 of 64 to 48, so the model got better at the maths as well.

??? note "The numbers"

    64 held-out MATH problems (levels 3-5), 8 answers each at temperature
    1.0, in per cent.

    | Step | Correct | Boxed final answer |
    | ---: | ---: | ---: |
    | 0 | 5.9 | 29 |
    | 50 | 26.8 | 61 |
    | 100 | 45.7 | 96 |
    | 150 | 49.6 | 99 |
    | 200 | 49.8 | 99 |
    | 250 | 49.0 | 98 |
    | 300 | 52.3 | 99 |
    | 350 | 51.8 | 99 |

The finished model was then scored at SimpleRL-Zoo's own settings, beside the
base it started from and SimpleRL-Zoo's published checkpoint:

<figure markdown="span">
  ![Dot plot of accuracy on two benchmarks. GSM8K, first 200 problems: base 36.3%, our GRPO run 68.2%, SimpleRL-Zoo's checkpoint 74.5% against its published 74.4%. MATH-500: base 14.4%, our GRPO run 54.7%, SimpleRL-Zoo's checkpoint 57.2% against its published 59.0%.](assets/figures/grpo-simplerl-zoo-light.svg#only-light)
  ![Dot plot of accuracy on two benchmarks. GSM8K, first 200 problems: base 36.3%, our GRPO run 68.2%, SimpleRL-Zoo's checkpoint 74.5% against its published 74.4%. MATH-500: base 14.4%, our GRPO run 54.7%, SimpleRL-Zoo's checkpoint 57.2% against its published 59.0%.](assets/figures/grpo-simplerl-zoo-dark.svg#only-dark)
</figure>

Two things show here. SimpleRL-Zoo's checkpoint, measured on this stack,
lands within 0.1 points (GSM8K) and 1.8 points (MATH-500) of its published
numbers, so the measurement can be trusted. And our run recovers 84% of
their gain over the base on GSM8K and 94% on MATH-500, from about a
twentieth of their rollouts.

??? note "The numbers"

    Accuracy in per cent at temperature 1.0 and top-p 0.95: mean ± standard
    deviation over three runs. GSM8K is scored on the first 200 of its 1,319
    test problems.

    | Model | GSM8K | MATH-500 |
    | --- | ---: | ---: |
    | Qwen2.5-1.5B base | 36.3 ± 4.2 | 14.4 ± 2.6 |
    | Our GRPO run | 68.2 ± 1.4 | 54.7 ± 0.4 |
    | SimpleRL-Zoo's checkpoint | 74.5 ± 1.8 | 57.2 ± 0.9 |
    | SimpleRL-Zoo's published score | 74.4 | 59.0 |
