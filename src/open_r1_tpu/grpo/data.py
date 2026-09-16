"""GRPO prompt-and-gold-answer dataset loading.

Unlike :mod:`open_r1_tpu.sft.data` (SFT: full conversations, packed into
fixed-shape token windows with an assistant-only loss mask), a GRPO training
batch needs only a rendered prompt string for the rollout sampler plus enough
of the source row for the reward functions in
:mod:`open_r1_tpu.grpo.rewards` to judge the sampler's own completions --
here, the bare gold answer. There is nothing to tokenize or pack here: Tunix's
rollout takes prompt strings and tokenizes them itself.

The recipe (``recipes/OpenR1-Distill-Qwen2.5-Math-1.5B/grpo``) reads
``open-r1/DAPO-Math-17k-Processed``'s ``en`` config: one ``prompt`` string
(the bare problem, no instruction wrapper) and one ``solution`` string (a
bare integer) per row, named via ``dataset.question_column`` and
``dataset.answer_column`` so any two-column prompt/gold corpus can be
substituted by override. It is deliberately *not* the SFT corpus: the actor
has already imitated those traces for six epochs, and GRPO's signal is the
within-group variance of rollouts, which memorised prompts collapse.
"""

from __future__ import annotations

from typing import Any

from open_r1_tpu.core.config import read_prompt_file


def render_prompt(question: str, tokenizer: Any, *, system_prompt: str | None) -> str:
    """Render one user turn with the generation prompt open, as raw text.

    ``add_generation_prompt=True`` and ``tokenize=False`` mirror how the
    eval stack's own chat completions are built (the served model sees
    exactly this shape at evaluation time too) and how
    ``examples/grpo_gemma.ipynb``'s ``TEMPLATE.format(...)`` renders its
    prompts, just via the tokenizer's own chat template instead of a
    hand-written one.
    """
    rendered = _render(question, tokenizer, system_prompt=system_prompt, tokenize=False)
    if not isinstance(rendered, str):
        raise ValueError("chat template did not return a rendered string")
    return rendered


def _render(
    question: str, tokenizer: Any, *, system_prompt: str | None, tokenize: bool
) -> Any:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=tokenize, add_generation_prompt=True
    )


def _prompt_token_count(
    question: str, tokenizer: Any, *, system_prompt: str | None
) -> int:
    # A second apply_chat_template call rather than tokenizer.encode() on the
    # already-rendered text: the tokenizer here may be Tunix's own adapter
    # (tunix.generate.tokenizer_adapter), which apply_chat_template is known
    # to support (model.tokenizing's render_ids relies on the same call) but
    # whose plain encode() is not exercised anywhere else in this project.
    ids = _render(question, tokenizer, system_prompt=system_prompt, tokenize=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def build_row_encoder(
    tokenizer: Any,
    *,
    question_column: str,
    answer_column: str,
    system_prompt: str | None,
    max_prompt_length: int | None,
) -> Any:
    """Build the per-row transform ``load_grpo_prompts`` hands to Grain.

    A separate function -- rather than a closure inline in
    ``load_grpo_prompts`` -- so it can be unit-tested directly against a fake
    tokenizer and plain dict rows, without a real HF ``datasets`` object or
    Grain installed. Returns ``None`` for a row Grain should drop: missing a
    gold answer, or (when ``max_prompt_length`` is set) too long for the
    rollout's prompt budget.
    """

    def to_record(row: dict[str, Any]) -> dict[str, Any] | None:
        question = str(row[question_column])
        if max_prompt_length is not None:
            token_count = _prompt_token_count(
                question, tokenizer, system_prompt=system_prompt
            )
            if token_count > int(max_prompt_length):
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
    """Wrap a Hugging Face split in a lazy, batched Grain prompt dataset.

    The seam ``load_grpo_prompts`` calls through (rather than building this
    inline), so a test can monkeypatch it exactly as
    ``tests/test_grpo_data.py`` does for ``grpo.data.build_grain_prompt_batches``
    and exercise the row-shaping and splitting logic without Grain
    installed.
    """
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
    """Load a Hugging Face math corpus as Tunix-ready GRPO prompt batches.

    Returns ``(train_ds, eval_ds)``, the second ``None`` unless
    ``dataset.eval_fraction`` is set -- mirroring
    ``sft.data.load_reasoning_datasets``'s split convention, minus the
    tokenization/packing that only SFT needs.
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

    system_prompt = (
        read_prompt_file(config["system_prompt_file"])
        if config.get("system_prompt_file") is not None
        else None
    )
    to_record = build_row_encoder(
        tokenizer,
        question_column=question_column,
        answer_column=answer_column,
        system_prompt=system_prompt,
        max_prompt_length=config.get("max_prompt_length"),
    )
    common = {
        "to_record": to_record,
        "batch_size": int(config["batch_size"]),
        "seed": int(config.get("seed", 42)),
    }

    train_ds = build_grain_prompt_batches(
        train_source,
        shuffle=True,
        num_epochs=int(config.get("num_train_epochs", 1)),
        **common,
    )
    eval_ds = (
        build_grain_prompt_batches(eval_source, shuffle=False, num_epochs=1, **common)
        if eval_source is not None
        else None
    )
    return train_ds, eval_ds
