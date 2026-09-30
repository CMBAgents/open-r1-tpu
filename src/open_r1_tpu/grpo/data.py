"""GRPO prompt-and-gold-answer dataset loading.

A GRPO batch needs only the rendered prompt string, which Tunix's rollout
tokenizes itself, plus the columns the reward functions in
:mod:`open_r1_tpu.grpo.rewards` judge completions against: the question and
the gold answer, read from ``dataset.question_column`` and
``dataset.answer_column``.
"""

from __future__ import annotations

from typing import Any

from open_r1_tpu.core.config import read_prompt_file
from open_r1_tpu.model.tokenizing import render_ids, user_turn


def render_prompt(question: str, tokenizer: Any, *, system_prompt: str | None) -> str:
    """Render one user turn as text, with the assistant turn opened."""
    rendered = tokenizer.apply_chat_template(
        user_turn(question, system_prompt), tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str):
        raise ValueError("chat template did not return a rendered string")
    return rendered


def build_row_encoder(
    tokenizer: Any,
    *,
    question_column: str,
    answer_column: str,
    system_prompt: str | None,
    max_prompt_length: int | None,
) -> Any:
    """Build the per-row transform ``load_grpo_prompts`` hands to Grain.

    The transform returns ``None`` for a row whose rendered prompt is longer
    than ``max_prompt_length`` tokens, when that is set, so Grain drops it.
    """

    def to_record(row: dict[str, Any]) -> dict[str, Any] | None:
        question = str(row[question_column])
        if max_prompt_length is not None:
            prompt_ids = render_ids(
                tokenizer,
                user_turn(question, system_prompt),
                add_generation_prompt=True,
            )
            if len(prompt_ids) > int(max_prompt_length):
                return None
        prompt = render_prompt(question, tokenizer, system_prompt=system_prompt)
        return {
            "prompts": prompt,
            "question": question,
            "answer": str(row[answer_column]),
        }

    return to_record


def build_grain_prompt_batches(
    source: Any,
    to_record: Any,
    *,
    batch_size: int,
    num_epochs: int,
    shuffle: bool,
    seed: int,
) -> Any:
    """Wrap a Hugging Face split in a lazy, batched Grain prompt dataset."""
    from grain import python as grain

    dataset = grain.MapDataset.source(source)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if num_epochs > 1:
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(to_record)
    dataset = dataset.filter(lambda record: record is not None)
    dataset = dataset.to_iter_dataset()
    return dataset.batch(batch_size=batch_size, drop_remainder=True)


def load_grpo_prompts(config: dict[str, Any], tokenizer: Any) -> tuple[Any, Any]:
    """Load a prompt corpus as Tunix-ready GRPO prompt batches.

    Rows missing a question or a gold answer are dropped. Returns
    ``(train_ds, eval_ds)``, the second ``None`` unless
    ``dataset.eval_fraction`` is set.
    """
    from datasets import load_dataset

    load_kwargs = {"split": config.get("train_split", "train")}
    if config.get("data_files") is not None:
        load_kwargs["data_files"] = config["data_files"]
    raw = load_dataset(config["name"], config.get("config"), **load_kwargs)

    question_column = config.get("question_column", "problem")
    answer_column = config.get("answer_column", "answer")
    raw = raw.filter(
        lambda row: bool(row.get(question_column)) and bool(row.get(answer_column))
    )

    max_examples = config.get("max_examples")
    if max_examples is not None:
        raw = raw.select(range(min(int(max_examples), len(raw))))

    eval_fraction = float(config.get("eval_fraction", 0.0))
    if eval_fraction:
        split = raw.train_test_split(
            test_size=eval_fraction, seed=int(config.get("seed", 42))
        )
        train_source, eval_source = split["train"], split["test"]
        eval_max_examples = config.get("eval_max_examples")
        if eval_max_examples is not None:
            eval_source = eval_source.select(
                range(min(int(eval_max_examples), len(eval_source)))
            )
    else:
        train_source, eval_source = raw, None

    to_record = build_row_encoder(
        tokenizer,
        question_column=question_column,
        answer_column=answer_column,
        system_prompt=read_prompt_file(config.get("system_prompt_file")),
        max_prompt_length=config.get("max_prompt_length"),
    )
    common = {"to_record": to_record, "seed": int(config.get("seed", 42))}
    batch_size = int(config["batch_size"])
    # Tunix scores each eval batch in one pass without micro-batching, so a
    # large-vocabulary model can need a smaller eval batch than its training one.
    eval_batch_size = int(config.get("eval_batch_size") or batch_size)

    train_ds = build_grain_prompt_batches(
        train_source,
        shuffle=True,
        num_epochs=int(config.get("num_train_epochs", 1)),
        batch_size=batch_size,
        **common,
    )
    eval_ds = (
        build_grain_prompt_batches(
            eval_source,
            shuffle=False,
            num_epochs=1,
            batch_size=eval_batch_size,
            **common,
        )
        if eval_source is not None
        else None
    )
    return train_ds, eval_ds
