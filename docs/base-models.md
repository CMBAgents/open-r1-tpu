# Base models

Every run starts from a *base model*: an open language model that has been
pretrained, and nothing more. This page explains what one is, what is inside
it, and how to download and prepare one.

## What is a base model?

A language model reads text as a sequence of *tokens*, pieces of words, and
is trained to predict the next one. *Pretraining* does this over trillions of
tokens of web pages, books and code, which takes thousands of accelerators
for weeks. The result is a base model.

A base model knows a great deal, but it has only ever continued documents. It
has not been taught to answer questions, to take turns in a conversation, or
to stop when it has finished. Given the start of a sentence, it carries on:

```bash
python scripts/complete_tpu.py \
  --model-path models/Qwen2.5-Math-1.5B --model-name qwen2.5-math-1.5b \
  "The capital of France is"
```

Given a question, it may answer it, but it may as easily write a second
question, or keep going until it runs out of room. *Post-training* turns it
into an assistant; see [What is post-training?](post-training.md).

Models with *Instruct* or *Chat* in their names are base models that someone
has already post-trained. This pipeline starts from base models, so that the
post-training is its own and can be measured.

## Tokens and the chat template

A conversation has to become a single sequence of tokens before a model can
read it. The tokeniser's *chat template* does this, marking where each turn
starts and ends with *special tokens*. Qwen's template renders a system
prompt, a question and the start of the reply like this:

```text
<|im_start|>system
Solve the maths problem. Think it through step by step between <think> and </think>, then give the final answer as a number in \boxed{}.<|im_end|>
<|im_start|>user
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?<|im_end|>
<|im_start|>assistant
```

The model writes its reply after the last line and ends it with
`<|im_end|>`. Two tokens can stop a Qwen model, and the difference matters:

| Token | Learnt in | Means |
| --- | --- | --- |
| `<|endoftext|>` | Pretraining | The end of a document |
| `<|im_end|>` | Post-training | The end of a turn in a conversation |

A base model has seen `<|endoftext|>` between billions of documents, but it
has never been taught to end a turn. Writing `<|im_end|>` at the right moment
is one of the first things SFT teaches it. It is also why the evaluation
[checks](evaluation.md#run) that a model's `generation_config.json` lists
`<|im_end|>` as a stop token: a server that stopped only at `<|endoftext|>`
would let every reply run on.

## What is in a model directory

A model on the [Hugging Face Hub](https://huggingface.co) is a directory of
files:

| File | Holds |
| --- | --- |
| `model.safetensors` | The weights: for a 1.5B model, 1.5 billion numbers |
| `config.json` | The network's shape: layers, widths, attention heads, context length |
| `tokenizer.json`, `tokenizer_config.json` | How text is split into tokens, the special tokens, and the chat template |
| `generation_config.json` | Defaults for generating text, including the stop tokens |

Training writes its finished model in the same format, so anything that reads
a Hub model can read it.

## The models used here

| Model | Used for |
| --- | --- |
| [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B) | [SFT](sft.md). A 1.5B base pretrained further on maths. DeepSeek distilled R1 into this same base to make DeepSeek-R1-Distill-Qwen-1.5B, so the two compare directly. |
| [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) | [RL](grpo.md), reproducing SimpleRL-Zoo, which starts from this base |
| [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) | The [tutorials](tutorials.md): small enough to train in full on one chip |
| [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) | Not a base: the reference model that [benchmarking](benchmarking.md) checks itself against |

Training and export support the Qwen2 and Qwen3 families.

## Downloading a base model

The Hub's `hf` command is installed with the environment. Create an access
token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
and set it as `HF_TOKEN` in `~/.open-r1-tpu.env` (see [Install](install.md#configure)).
Tunix's downloader needs it even for public models. Then download into
`models/`, which git ignores:

```bash
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B
```

A recipe names the model in its `model` section:

```yaml
model:
  model_name: qwen2.5-math-1.5b        # Tunix's name for the architecture
  model_id: Qwen/Qwen2.5-Math-1.5B     # the Hub repository
  model_source: local
  model_path: models/Qwen2.5-Math-1.5B # the directory downloaded above
```

The shipped recipes load from a local directory (`model_source: local`)
rather than from the Hub: Tunix's Hub loader contacts the Hub even when the
weights are already on disk.

**Keep a copy in your bucket.** Copy each downloaded model up once, and later
VMs can copy it down instead of downloading it again:

```bash
gcloud storage rsync --recursive models/Qwen2.5-Math-1.5B \
  gs://YOUR_BUCKET/models/Qwen2.5-Math-1.5B
```

## Preparing a model for long reasoning

Reasoning traces are long: the SFT recipe trains on windows of 26,624 tokens.
Qwen2.5-Math-1.5B was pretrained on at most 4,096, and a model asked to read
far beyond its pretrained length loses track of where each token is.

The model knows a token's position through *RoPE* (rotary position
embeddings), which rotates each token's representation by an angle that
grows with its position. The base frequency θ (`rope_theta`) sets how fast
the angle turns; a larger θ turns more slowly, so far-apart positions remain
distinguishable. Open-R1 raised θ from 10,000 to 300,000 for long traces, and
so does this project: at 10,000 the 26k-token window ended in repetition
loops, and at 300,000, with a higher learning rate, MATH-500 rose from 48% to
83%.

`scripts/set_rope.py` writes the new θ and context length into a downloaded
model's `config.json`; the weights are unchanged, and training adapts them:

```bash
hf download Qwen/Qwen2.5-Math-1.5B --local-dir models/Qwen2.5-Math-1.5B-RoPE-300k
python scripts/set_rope.py models/Qwen2.5-Math-1.5B-RoPE-300k \
  --rope-theta 300000 --max-position-embeddings 32768
```

The recipe must set `model.rope_theta` to the same value, because Tunix
reads θ from the recipe rather than from `config.json`.
