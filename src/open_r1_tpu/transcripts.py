"""Periodic free-running transcripts for qualitative inspection during SFT.

Teacher-forced loss cannot show whether the trained model closes its reasoning
trace, stops at EOS, or degenerates into repetition, because every scored token
is conditioned on ground truth. Sampling a fixed prompt set at a fixed interval
exposes those failures while a run is still cheap to abandon.

Sampling is not free here the way rollouts are in GRPO: it adds an autoregressive
decode, a second XLA compilation, and a KV cache to a device whose memory profile
was validated without them. It is therefore opt-in, bounded, and never fatal --
a failed sample logs a warning and training continues.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

# Short, verifiable prompts. The point is comparability across steps, not
# coverage, so keep the set small and fixed.
DEFAULT_PROMPTS = (
    "What is the remainder when 2^100 is divided by 7?",
    "A train travels 60 km in 45 minutes. What is its average speed in km/h?",
    "Is 91 a prime number? Show your reasoning.",
)


def resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``training.transcripts`` and apply defaults."""
    training = config["training"]
    dataset = config["dataset"]
    raw = training.get("transcripts") or {}

    max_new_tokens = int(raw.get("max_new_tokens", 256))
    prompts = [str(prompt) for prompt in (raw.get("prompts") or DEFAULT_PROMPTS)]
    output_path = raw.get("output_path") or str(
        Path(training["metrics_log_dir"]).parent / "transcripts.jsonl"
    )
    return {
        "enabled": bool(raw.get("enabled", False)),
        "every_n_steps": int(raw.get("every_n_steps", 500)),
        "max_new_tokens": max_new_tokens,
        "temperature": float(raw.get("temperature", 0.0)),
        "seed": int(raw.get("seed", dataset.get("seed", 42))),
        # The cache holds the rendered prompt as well as the completion.
        # An explicit null in the recipe means "derive it", so treat a missing
        # key and a null value the same way.
        "cache_size": int(
            raw.get("cache_size") or int(dataset["max_length"]) + max_new_tokens
        ),
        "log_to_wandb": bool(raw.get("log_to_wandb", True)),
        "prompts": prompts,
        "output_path": output_path,
        "system_prompt": dataset.get("system_prompt"),
        "reasoning_start": dataset.get("reasoning_start", "<think>"),
        "reasoning_end": dataset.get("reasoning_end", "</think>"),
    }


def should_sample(step: int, every_n_steps: int) -> bool:
    """Sample on the interval only; step 0 has nothing trained to show."""
    if every_n_steps <= 0:
        return False
    return step > 0 and step % every_n_steps == 0


def render_prompt(tokenizer: Any, prompt: str, system_prompt: str | None) -> str:
    """Render one user turn the way training rendered its supervised prefix."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_record(
    step: int,
    prompt: str,
    completion: str,
    *,
    reasoning_start: str,
    reasoning_end: str,
    max_new_tokens: int,
    generated_tokens: int | None = None,
) -> dict[str, Any]:
    """Summarize one sample, flagging the failures loss cannot show."""
    start_index = completion.find(reasoning_start)
    end_index = completion.find(reasoning_end)
    return {
        "step": step,
        "prompt": prompt,
        "completion": completion,
        "completion_chars": len(completion),
        "generated_tokens": generated_tokens,
        "has_reasoning_start": start_index != -1,
        "has_reasoning_end": end_index != -1,
        # An unbalanced or missing trace means the model never learned to close
        # its reasoning, which is invisible under teacher forcing.
        "reasoning_balanced": start_index != -1
        and end_index != -1
        and end_index > start_index,
        # A completion that used the whole budget was probably cut off, so its
        # missing closing tag is inconclusive rather than a real failure.
        "hit_token_cap": generated_tokens is not None
        and generated_tokens >= max_new_tokens,
    }


def write_records(output_path: str, records: list[dict[str, Any]]) -> None:
    """Append records as JSONL, the durable copy that survives a W&B outage."""
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_records_to_wandb(records: list[dict[str, Any]], step: int) -> None:
    """Log transcripts straight to the W&B run.

    Tunix's metrics path only carries stepped scalars, so text cannot travel
    through it. Logging here with the real training step keeps the ordering W&B
    requires, which is what the scalar filter exists to protect.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return

    columns = [
        "step",
        "prompt",
        "completion",
        "generated_tokens",
        "reasoning_balanced",
        "hit_token_cap",
    ]
    table = wandb.Table(
        columns=columns,
        data=[[record.get(column) for column in columns] for record in records],
    )
    wandb.log({"samples/transcripts": table}, step=step)


def sample_transcripts(
    sampler: Any,
    tokenizer: Any,
    settings: dict[str, Any],
    step: int,
) -> list[dict[str, Any]]:
    """Generate one completion per configured prompt and summarize each."""
    rendered = [
        render_prompt(tokenizer, prompt, settings["system_prompt"])
        for prompt in settings["prompts"]
    ]
    output = sampler(
        input_strings=rendered,
        max_generation_steps=int(settings["max_new_tokens"]),
        temperature=float(settings["temperature"]),
        seed=int(settings["seed"]),
    )
    completions = list(output.text)
    token_counts: list[int | None] = [None] * len(completions)
    tokens = getattr(output, "tokens", None)
    if tokens is not None:
        token_counts = [len(row) for row in tokens]

    return [
        build_record(
            step,
            prompt,
            completion,
            reasoning_start=settings["reasoning_start"],
            reasoning_end=settings["reasoning_end"],
            max_new_tokens=int(settings["max_new_tokens"]),
            generated_tokens=count,
        )
        for prompt, completion, count in zip(
            settings["prompts"], completions, token_counts
        )
    ]


def create_training_hooks(model: Any, tokenizer: Any, settings: dict[str, Any]) -> Any:
    """Build the Tunix training hook that samples on the configured interval."""
    from tunix.generate import sampler as sampler_lib
    from tunix.sft import hooks

    def build_sampler() -> Any:
        model_config = getattr(model, "config", None)
        cache_config = sampler_lib.CacheConfig(
            cache_size=int(settings["cache_size"]),
            num_layers=int(getattr(model_config, "num_layers")),
            num_kv_heads=int(getattr(model_config, "num_kv_heads")),
            head_dim=int(getattr(model_config, "head_dim")),
        )
        return sampler_lib.Sampler(
            transformer=model,
            tokenizer=tokenizer,
            cache_config=cache_config,
        )

    class TranscriptHooks(hooks.TrainingHooks):
        """Sample a fixed prompt set every N steps, without ever failing a run."""

        def __init__(self) -> None:
            self._sampler: Any = None
            self._disabled = False

        def on_train_start(self, train_ctx: Any) -> None:
            pass

        def on_train_end(self, train_ctx: Any) -> None:
            pass

        def on_train_step_start(self, train_ctx: Any) -> None:
            pass

        def on_eval_step_start(self, train_ctx: Any) -> None:
            pass

        def on_eval_step_end(self, train_ctx: Any, eval_loss: float) -> None:
            pass

        def on_train_step_end(
            self, train_ctx: Any, train_step: int, train_loss: float
        ) -> None:
            if self._disabled:
                return
            if not should_sample(train_step, int(settings["every_n_steps"])):
                return
            try:
                # Built on first use so the decode compilation is only paid by
                # runs that actually sample.
                if self._sampler is None:
                    LOGGER.info("Building transcript sampler at step %d", train_step)
                    self._sampler = build_sampler()
                records = sample_transcripts(
                    self._sampler, tokenizer, settings, train_step
                )
                write_records(settings["output_path"], records)
                if settings["log_to_wandb"]:
                    log_records_to_wandb(records, train_step)
                unbalanced = sum(
                    1
                    for record in records
                    if not record["reasoning_balanced"] and not record["hit_token_cap"]
                )
                LOGGER.info(
                    "Wrote %d transcripts at step %d to %s (%d without a closed "
                    "reasoning trace)",
                    len(records),
                    train_step,
                    settings["output_path"],
                    unbalanced,
                )
            except Exception:  # noqa: BLE001 - inspection must never kill a run
                # Disable rather than retry: a sampler that OOMs or fails to
                # compile will do so at every interval, and the training run
                # matters more than the transcripts.
                self._disabled = True
                LOGGER.warning(
                    "Transcript sampling failed at step %d and is now disabled "
                    "for the rest of this run; training continues.",
                    train_step,
                    exc_info=True,
                )

    return TranscriptHooks()
