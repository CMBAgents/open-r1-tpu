#!/usr/bin/env python3
"""Interactive chat REPL against a rowanai merged export.

Every one of rowanai's 2,269 training examples is exactly one user turn plus
one assistant turn (verified: {len(row["messages"]) for row in dataset} ==
{2}, no exceptions). The model has no training signal for a second user turn,
and conditioning on one is out-of-distribution: it stops responding to the
new message's actual content and just continues generating in whatever
register the first turn established. So this REPL resets history after every
reply by default - each message is sent as a fresh single turn, matching the
training distribution. Pass --multiturn to opt into keeping history anyway
(useful for seeing the degradation, not for getting usable replies).

Type /reset to clear history early, /exit or Ctrl-D to quit.

Usage:
    python3 chat_rowanai_interactive.py [model_dir] [--multiturn]
Defaults to the selected coherence checkpoint, artifacts/rowanai-iterate/v4/merged.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = "artifacts/rowanai-iterate/v4/merged"


def main():
    args = [a for a in sys.argv[1:] if a != "--multiturn"]
    multiturn = "--multiturn" in sys.argv[1:]
    model_dir = args[0] if args else DEFAULT_MODEL_DIR

    print(f"Loading {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()
    if multiturn:
        mode = "multi-turn (opted in; expect degraded replies past turn 1)"
    else:
        mode = "single-turn (default; each message starts fresh)"
    print(f"Ready, mode: {mode}. Type /reset to clear history, /exit to quit.\n")

    history = []
    while True:
        try:
            user_message = input("you> ").strip()
        except EOFError:
            print()
            break
        if not user_message:
            continue
        if user_message in ("/exit", "/quit"):
            break
        if user_message == "/reset":
            history = []
            print("(history cleared)")
            continue

        history.append({"role": "user", "content": user_message})
        inputs = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                tokenizer=tokenizer,
                stop_strings=["<|im_end|>"],
                pad_token_id=tokenizer.pad_token_id,
            )
        reply_ids = output[0][inputs["input_ids"].shape[-1] :]
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
        print(f"rowanai> {reply}\n")

        if multiturn:
            history.append({"role": "assistant", "content": reply})
        else:
            history = []


if __name__ == "__main__":
    main()
