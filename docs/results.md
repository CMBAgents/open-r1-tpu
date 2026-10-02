# Results

Each model this project trains is compared with the published model it
reproduces, both measured on this project's evaluation suite (see
[Benchmarking](benchmarking.md)), so the comparison is like for like. The
numbers behind each chart are in the table under it.

## Distillation (SFT)

The [SFT recipe](sft.md) distils DeepSeek-R1's reasoning into
Qwen2.5-Math-1.5B. DeepSeek did the same with their own data to make
DeepSeek-R1-Distill-Qwen-1.5B, so that is the model to compare with.

<figure markdown="span">
  ![Grouped bar chart. MATH-500: our distilled model 83.7%, DeepSeek-R1-Distill-Qwen-1.5B 78.6%. AIME 2024: 24.3% and 30.6%. AIME 2024, majority of 64 runs: 43.3% and 53.3%. GPQA-Diamond: 22.7% and 28.3%.](assets/figures/sft-vs-deepseek-light.svg#only-light)
  ![Grouped bar chart. MATH-500: our distilled model 83.7%, DeepSeek-R1-Distill-Qwen-1.5B 78.6%. AIME 2024: 24.3% and 30.6%. AIME 2024, majority of 64 runs: 43.3% and 53.3%. GPQA-Diamond: 22.7% and 28.3%.](assets/figures/sft-vs-deepseek-dark.svg#only-dark)
</figure>

Our distilled model scores five points higher on MATH-500, the kind of
competition maths it was trained on, and lower on the harder AIME 2024 and
on GPQA-Diamond's science questions. On GPQA it closed its reasoning block in
only 27% of replies, against 98% for DeepSeek's model: away from competition
maths, it often gives an answer without finishing its reasoning.

??? note "The numbers"

    Accuracy in per cent with DeepSeek's settings: a 32,768-token budget,
    temperature 0.6 and no system prompt. Mean ± standard deviation over 4
    runs (MATH-500, GPQA-Diamond) or 64 runs (AIME 2024); the majority vote
    is one number over all 64 runs.

    | Benchmark | Our distilled model | DeepSeek-R1-Distill-Qwen-1.5B |
    | --- | ---: | ---: |
    | MATH-500 | **83.7 ± 1.2** | 78.6 ± 0.3 |
    | AIME 2024 | 24.3 ± 5.3 | **30.6 ± 6.4** |
    | AIME 2024, majority of 64 | 43.3 | **53.3** |
    | GPQA-Diamond | 22.7 ± 0.4 | **28.3 ± 0.0** |

## Reinforcement learning (GRPO)

The [RL recipe](grpo.md) reproduces SimpleRL-Zoo: GRPO with a 1/0
correctness reward, starting from the Qwen2.5-1.5B base. Their trained
checkpoint is public, so both models are measured on the same suite.

<figure markdown="span">
  ![Grouped bar chart. GSM8K, first 200 problems: our GRPO model 68.2%, SimpleRL-Zoo's checkpoint 74.5%. MATH-500: 54.7% and 57.2%.](assets/figures/grpo-vs-simplerl-zoo-light.svg#only-light)
  ![Grouped bar chart. GSM8K, first 200 problems: our GRPO model 68.2%, SimpleRL-Zoo's checkpoint 74.5%. MATH-500: 54.7% and 57.2%.](assets/figures/grpo-vs-simplerl-zoo-dark.svg#only-dark)
</figure>

Our GRPO model is 6.3 points behind on GSM8K and 2.5 on MATH-500. The base
both started from scores 36.3 and 14.4 at these settings, so our run
recovers 84% and 94% of SimpleRL-Zoo's gain, with a LoRA adapter and a
sixty-fourth of their batch. SimpleRL-Zoo's checkpoint, measured here, lands
within 0.1 and 1.8 points of their published numbers.

??? note "The numbers"

    Accuracy in per cent with SimpleRL-Zoo's settings: temperature 1.0 and
    top-p 0.95. Mean ± standard deviation over three runs. GSM8K is scored on
    the first 200 of its 1,319 test problems.

    | Benchmark | Our GRPO model | SimpleRL-Zoo's checkpoint |
    | --- | ---: | ---: |
    | GSM8K | 68.2 ± 1.4 | **74.5 ± 1.8** |
    | MATH-500 | 54.7 ± 0.4 | **57.2 ± 0.9** |
