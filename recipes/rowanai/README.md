# Matched SFT comparison

Compare the local `rowanai` Llama base with `Qwen2.5-Math-1.5B-RoPE-300k`
on the same locally staged worked solutions. Models and data are private:
keep them in ignored `models/`, `data/`, and `artifacts/`, and stage only through
the private GCS bucket supplied in `GCS_BUCKET`; do not upload them to GitHub
or Hugging Face. The preparation manifest belongs beside private artifacts.

Both arms retain the successful full-SFT optimizer: Adam, LR 2e-4, warmup
3%, cosine floor 10%, betas 0.9/0.999, epsilon 1e-8, zero weight decay, gradient
clip 0.2, decoder rematerialization, flash attention, and assistant-only loss.
The target is one v6e chip with `[fsdp, tp] = [1, 1]`. TPU preflight and a
four-update smoke (including checkpoint and merged export) are required on an
idle one-chip VM before full training; CPU checks do not establish TPU memory
or compilation compatibility.

The matched protocol is:

- 2,269 examples in identical source order; Grain shuffle seed 42 in both arms.
- No packing: different tokenizers must not change which examples share a step.
- Microbatch 2, accumulation 8: 16 examples per optimizer update, 709 updates.
- Five dataset passes contain 11,345 examples; 11,344 form complete updates,
  so the final single example is unused in both arms.
- Length 3,072, overlength dropping, no truncation: the whole corpus was checked
  with both actual tokenizers (maximum lengths 2,800 and 2,811 respectively).
- Worked solutions have no reasoning tags; `require_reasoning_tags=false`
  preserves them verbatim, without inventing reasoning/answer boundaries.
- `rowanai` uses ChatML with its existing vocabulary: `<|im_start|>role\n`
  begins a message and `<|im_end|>\n` closes it; neither marker is a new token.
  No default system message is added to this arm.
- Qwen uses its original native template, including its default maths system
  message and existing special ChatML tokens. Both arms keep their own template
  for evaluation; the prompt formatting now intentionally differs.
- Eval loss and generation transcripts remain off as in the successful SFT;
  evaluate on independent questions, never treat these training rows as a test.
- Checkpoint every 100 updates, retain two, export at completion, separate
  artifact paths and W&B names. W&B source upload is disabled.

`rowanai` has 1,068,550,144 parameters, 12 Llama layers, 32k vocabulary, and
native RoPE theta 50,000. Its local loader reads topology from `config.json`
and uses the pinned Tunix Qwen2 kernels with parameter-free zero projection
biases, which implements the bias-free Llama equations while preserving flash
attention and segmentation. Its source checkpoint's 12 derived RoPE buffers
were checked against theta 50,000 and omitted from safetensors; learned tensors
are unchanged. Document EOS is 0 and PAD is 1, with no BOS inserted. ChatML turn endings
use seven ordinary tokens; `export.stop_strings` stores the full `<|im_end|>`
marker in the exported generation config, preserving document EOS. No tokens
are added or repurposed, and the embedding/output matrices are unchanged.
For Transformers generation, supply `tokenizer=tokenizer` so stop strings work;
for vLLM/OpenAI-compatible inference, pass `stop=["<|im_end|>"]` explicitly.
Never put the marker's first token (`<`, ID 29) in `eos_token_id`.
Qwen retains its public base weights, with only context settings changed to
RoPE theta 300,000 and 32,768 positions, and its original native chat template.

These bases also differ in architecture, parameter count, vocabulary, and
pretraining, so this experiment compares the two supplied bases; it cannot
isolate the effect of historical-data restriction alone.

From the repository root on the training VM, with the standard pinned training
environment activated and `GCS_BUCKET` set:

```bash
./scripts/run_rowanai_comparison.sh rowanai stage
./scripts/run_rowanai_comparison.sh qwen stage
./scripts/run_rowanai_comparison.sh rowanai smoke
./scripts/run_rowanai_comparison.sh qwen smoke
# Inspect losses, saved checkpoints, merged exports and HBM before these:
./scripts/run_rowanai_comparison.sh rowanai train
./scripts/run_rowanai_comparison.sh qwen train
```

Run the arms sequentially, with identical hardware/software. The launcher does
not start or allocate a VM. Full-run checkpoints go directly to GCS; logs and
merged exports are synced after a successful run. If a run fails, save its
local log to GCS and record its exact command, step, last checkpoint and failure
before resuming. A re-run against an existing GCS checkpoint prefix resumes;
use a new prefix for an independent experiment.

## rowanai coherence follow-up (branch `rowanai`)

`config_rowanai.yaml` above (5 epochs, matched to Qwen for the comparison) is
undertrained for standalone chat use: greedy decoding degenerates into
repetition loops on most prompts. `config_rowanai_v{2,3,4}.yaml` sweep
training epochs and gradient clipping on the same base/data/tokenizer to fix
that, writing to `artifacts/rowanai-iterate/v{2,3,4}/` so the matched
comparison above is never touched. Findings, in order:

- v2 (10 epochs, `max_grad_norm 0.2`): removes almost all repetition loops
  seen at 5 epochs.
- v3 (20 epochs, same clip): a regression — loops reappear and off-domain
  replies get more garbled. The epoch/coherence relationship is non-monotonic.
- **v4 (10 epochs, `max_grad_norm 1.0`) is the best checkpoint found**: no
  long repetition loops across a 10-prompt in-domain/out-of-domain test set,
  lower held-out eval loss than v2 (2.66 vs 2.75), and at least one fully
  correct worked solution. Use `artifacts/rowanai-iterate/v4/merged` for any
  standalone rowanai chat/inference work, not the matched-comparison export.

Remaining limitation: prompts far from the pre-1905 worked-mathematics corpus
(general trivia, creative writing) still get short non-answers or a fluently
regurgitated but off-topic memorized passage rather than a real answer. That
looks like the corpus's narrow scope, not a fixable training hyperparameter.
