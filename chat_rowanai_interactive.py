#!/usr/bin/env python3
"""Interactive multi-turn chat REPL against a rowanai merged export.

History lives only in this process's memory for the life of the REPL, so
there is no stale-history-file bug like the old chat_rowanai.py had. Type
/reset to clear history, /exit or Ctrl-D to quit.

Usage:
    python3 chat_rowanai_interactive.py [model_dir]
Defaults to the selected coherence checkpoint, artifacts/rowanai-iterate/v4/merged.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = "artifacts/rowanai-iterate/v4/merged"


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
    print(f"Loading {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()
    print("Ready. Type /reset to clear history, /exit to quit.\n")

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
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
