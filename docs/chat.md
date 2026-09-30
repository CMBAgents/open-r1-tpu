# Chat and completion on the TPU

Both scripts load a local Qwen2 or Qwen3 directory with Tunix and never
contact the Hub. Tunix needs the model's name, such as `qwen2.5-math-1.5b`
(the `model.model_name` in its recipe). It is read from `config.json`'s
`_name_or_path`, but Hub downloads and this project's exports have none, so
pass `--model-name`; with `--recipe`, chat takes it from the recipe.

Both use every visible TPU chip, so do not run them beside a training job or
an evaluation. The first reply compiles the decode path and is slower than
the rest.

## Chat

`chat_tpu.py` applies the tokenizer's chat template. Point it at a merged
export:

```bash
python scripts/chat_tpu.py \
  --model-path artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged \
  --model-name qwen2.5-math-1.5b
```

Or pass an SFT recipe to talk to a run before it finishes. The latest
checkpoint is restored on top of the recipe's local base (`model.model_path`,
or `model.model_download_path` for a Hub model), or on `--model-path`. A LoRA
recipe's adapters are restored with the rank, alpha and target modules from
the recipe, since a mismatch produces fluent nonsense rather than an error.

```bash
python scripts/chat_tpu.py \
  --recipe recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
```

`--step` picks an earlier checkpoint and `--checkpoint-dir` finds moved ones.
The restored step is printed on load; only every
`save_interval_steps`-th step is written and only `max_to_keep` are kept, so
asking for a missing step lists the ones that exist.

By default no system message is sent, so the chat template supplies its own
if it has one (Qwen2.5-Math's asks the model to reason step by step and box
its answer). To talk to a model as it was trained, pass the prompt it was
trained with, for example
`--system-prompt "$(cat recipes/Qwen2.5-Math-1.5B/system_prompt.txt)"`.
Sampling defaults to temperature 0.6 and top-p 0.95; `--temperature 0` is
greedy.

Two details match what training teaches:

- **Replies stop at `<|im_end|>`**, the chat template's turn end, as well as
  at the base model's EOS, `<|endoftext|>`. Stopping only at the EOS, the model
  would write past its reply and into your next turn.
- **An empty `<think></think>` block is hidden.** Qwen3's template adds one to
  every assistant turn without reasoning, so models trained on such data emit
  it before each answer.

## Raw completion

`complete_tpu.py` continues raw text with no chat template, system prompt or
history, which shows a base model's own next-token behaviour:

```bash
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B
python scripts/complete_tpu.py \
  --model-path models/Qwen2.5-Math-1.5B --model-name qwen2.5-math-1.5b \
  "The capital of France is"
```

Omit the prompt for an interactive loop. Decoding is greedy and stops at the
tokenizer's EOS (`<|endoftext|>` for Qwen base models) or after
`--max-new-tokens` (default 100). It uses ordinary
masked attention rather than splash attention: the pinned Tunix sampler
left-pads prompts but does not give splash attention the segment IDs it needs
to ignore the padding.
