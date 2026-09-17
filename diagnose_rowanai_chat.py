#!/usr/bin/env python3
"""Diagnose the rowanai merged export's identical-output chat behaviour.

Standalone, no history file: each prompt is a single fresh turn. Prints raw
token ids for the templated prompt, the generated reply ids, and the top-5
next-token logits at the first generation step, so we can see whether the
model is actually conditioning on the prompt at all.
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_DIR = "artifacts/rowanai-comparison/rowanai/merged"

PROMPTS = [
    "What is 2+3?",
    "What is Newton's third law of motion?",
    "Describe the colour of the sky.",
    "Prove that the sum of the angles of a triangle is equal to two right angles.",
    "Solve $x^2 - 5x + 6 = 0$.",
    "Prove that the diagonals of a parallelogram bisect each other.",
    "Solve $2x^2 + 3x - 5 = 0$.",
    "Find the sum of the first $n$ terms of the series $1 + 2 + 3 + \\cdots + n$.",
    "What is the capital of France?",
    "Write a short poem about autumn.",
]


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL_DIR
    quick = len(sys.argv) > 2 and sys.argv[2] == "--quick"
    print(f"MODEL_DIR={model_dir} quick={quick}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16)
    model.eval()

    print(
        f"eos_token_id={tokenizer.eos_token_id} pad_token_id={tokenizer.pad_token_id}"
    )
    print(
        f"decode([0])={tokenizer.decode([0])!r} decode([1])={tokenizer.decode([1])!r}"
    )

    for prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        input_ids = inputs["input_ids"]
        print("=" * 70)
        print(f"PROMPT: {prompt!r}")
        print(f"templated text: {tokenizer.decode(input_ids[0])!r}")
        print(f"input_ids ({input_ids.shape[-1]} tokens): {input_ids[0].tolist()}")

        if not quick:
            with torch.no_grad():
                first_logits = model(**inputs).logits[0, -1]
            top5 = torch.topk(first_logits, 5)
            print("top-5 next-token logits at first generation step:")
            for val, idx in zip(
                top5.values.tolist(), top5.indices.tolist(), strict=True
            ):
                print(f"  id={idx} tok={tokenizer.decode([idx])!r} logit={val:.3f}")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                tokenizer=tokenizer,
                stop_strings=["<|im_end|>"],
                pad_token_id=tokenizer.pad_token_id,
            )
        reply_ids = output[0][input_ids.shape[-1] :]
        print(f"reply_ids: {reply_ids.tolist()}")
        reply_text = tokenizer.decode(reply_ids, skip_special_tokens=False)
        print(f"reply (raw, with specials): {reply_text!r}")

        if not quick:
            with torch.no_grad():
                output_rp = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    tokenizer=tokenizer,
                    stop_strings=["<|im_end|>"],
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=1.3,
                    no_repeat_ngram_size=3,
                )
            reply_ids_rp = output_rp[0][input_ids.shape[-1] :]
            print(
                f"reply w/ repetition_penalty=1.3, no_repeat_ngram_size=3: "
                f"{tokenizer.decode(reply_ids_rp, skip_special_tokens=False)!r}"
            )
        sys.stdout.flush()


if __name__ == "__main__":
    main()
