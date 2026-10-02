# Benchmarking

Training reports a loss, but the loss is measured *teacher-forced*: the
model predicts each token of a correct reply with every earlier token of that
reply in front of it. It never shows whether the model, writing on its own,
closes its reasoning, stops, or reaches the right answer. Benchmarking does:
the model answers a fixed set of problems with known answers, and each
answer is checked.

## The benchmarks

| Benchmark | Problems | Tests |
| --- | --- | --- |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 1,319 | School maths word problems |
| [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 500 | Competition maths, from algebra to number theory |
| AIME 2024 and 2025 | 30 each | The American Invitational Mathematics Examination: hard, and so small that one problem is 3.3 points |
| [GPQA-Diamond](https://huggingface.co/datasets/Idavidrein/gpqa) | 198 | Graduate-level science questions, multiple choice |
| [IFEval](https://huggingface.co/datasets/google/IFEval) | 541 | Following explicit instructions, such as "answer in under 100 words" |
| LiveCodeBench | | Writing code that passes a problem's tests |

Every task's dataset, prompt and scoring come from
[LightEval](https://github.com/huggingface/lighteval), Hugging Face's
evaluation library, as Open-R1 uses them.

## How answers are scored

**pass@1** is the fraction of problems answered correctly, averaged over
samples. Maths answers are compared as mathematics, not as text, so
`\frac{1}{2}`, `0.5` and `1/2` all match.

**cons@n** (consensus) asks the same problem *n* times and takes the majority
answer, which is usually more accurate than any single attempt. DeepSeek
report both for AIME.

**Seeds.** A reasoning model samples its answers at a temperature, so two
runs on the same model give different scores, by 5 to 15 points on small
benchmarks ([arXiv 2504.07086](https://arxiv.org/abs/2504.07086)). Every
benchmark here runs several times and reports the mean and standard
deviation. How many runs depends on the benchmark's size: one run is ±8.3
points on AIME's 30 problems but ±1.6 on MATH-500's 500.

<figure markdown="span">
  ![64 runs each on AIME 2024: DeepSeek-R1-Distill-Qwen-1.5B ranges from 16.7% to 40.0% around a mean of 30.6%; our SFT model from 13.3% to 33.3% around a mean of 24.3%.](assets/figures/aime-runs-light.svg#only-light)
  ![64 runs each on AIME 2024: DeepSeek-R1-Distill-Qwen-1.5B ranges from 16.7% to 40.0% around a mean of 30.6%; our SFT model from 13.3% to 33.3% around a mean of 24.3%.](assets/figures/aime-runs-dark.svg#only-dark)
</figure>

## How an evaluation runs

1. **Preflight** checks, before anything takes the TPU, that the packages
   and the vLLM image are the pinned versions, that every task exists, and
   that the model will stop at `<|im_end|>`.
2. **Serving.** [vLLM](https://github.com/vllm-project/vllm), a fast
   inference server, loads the model onto one TPU chip inside a pinned Docker
   container and answers requests over an OpenAI-compatible API.
3. **Generation.** Each problem is turned into a prompt and sent to the
   server, many at a time.
4. **Scoring.** Each reply is scored with LightEval's metrics.
5. **Summary.** The scores are reduced to a mean and standard deviation per
   benchmark, with diagnostics: how often replies were cut off at the token
   budget (if often, the score is not trustworthy), how often the model used
   the format it was trained on, and how long its replies were.

Results stay on the VM as one JSONL file per seed and benchmark and a summary
file, and can be logged to Weights & Biases. With [Langfuse](langfuse.md)
connected, every prompt, reply and score can also be read in a browser.

Nothing on the host imports vLLM: it runs only inside the container, built
from `docker/vllm-tpu/` with every dependency pinned, so a result can be
reproduced with exactly the same software.

## Tiers

Evaluations come in tiers, from a quick check to a full report:

| Tier | Time | Contents | Use |
| --- | --- | --- | --- |
| 0, smoke | ~5 min | GSM8K, 200 problems, one greedy pass | Every run: catches a broken model |
| 1, core | ~1 h | MATH-500, three seeds | Whether a change helped |
| 2, headline | hours | AIME 2024 and 2025, ten seeds | Milestones: the numbers the field quotes |
| 3, regression | hours | IFEval and GPQA-Diamond | What training on maths alone cost |

## Checking the checker

A benchmark score is only comparable with a published one if it is measured
the same way. Two checks show this stack does:

- **SimpleRL-Zoo's** published checkpoint, scored here, lands within 0.1
  points (GSM8K) and 1.8 points (MATH-500) of its paper's numbers.
- **DeepSeek-R1-Distill-Qwen-1.5B** is set up to reproduce its model card's
  numbers, with DeepSeek's own token budget and sample counts. It matches the
  card on AIME, and scores about five points under it on MATH-500 and
  GPQA-Diamond.

<figure markdown="span">
  ![Accuracy on DeepSeek's protocol. MATH-500: our SFT model 83.7%, DeepSeek's card 83.9%, DeepSeek's model measured here 78.6%. AIME 2024: 24.3%, 28.9%, 30.6%. AIME 2024 majority of 64: 43.3%, 52.7%, 53.3%. GPQA-Diamond: 22.7%, 33.8%, 28.3%.](assets/figures/deepseek-card-light.svg#only-light)
  ![Accuracy on DeepSeek's protocol. MATH-500: our SFT model 83.7%, DeepSeek's card 83.9%, DeepSeek's model measured here 78.6%. AIME 2024: 24.3%, 28.9%, 30.6%. AIME 2024 majority of 64: 43.3%, 52.7%, 53.3%. GPQA-Diamond: 22.7%, 33.8%, 28.3%.](assets/figures/deepseek-card-dark.svg#only-dark)
</figure>

[Results](results.md) has every evaluation's numbers.

[Run an evaluation →](evaluation.md)
