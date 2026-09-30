"""The optional Langfuse path: trace every document of an evaluation.

With a tracing config, `evaluation.run` hands each run here. The recipe's
tasks are first synced into Langfuse datasets (`sync_recipe`): one dataset per
task, named `{task}@{fingerprint}` (plus `[:N]` when `eval.max_samples` caps
it, see `taskpack.dataset_name`), one item per document with a deterministic
id, so a re-sync upserts rather than duplicates. Then `dataset.run_experiment()`
drives generation and scoring per `(task, seed)`, with this project's task
function (`generate.make_task`) and a LightEval evaluator
(`lighteval_evaluator`), so each seed is its own run in Langfuse's comparison
view. The JSONL is written from the returned `ExperimentResult`, never read
back from Langfuse, so the summary does not depend on Langfuse staying up.

A document's `query` and `specific` are captured once, at sync, and scoring
rebuilds its `Doc` from them (`scoring.doc_from_item`) rather than calling the
prompt function again: `gpqa`'s shuffles its answer choices on every call.

`run_experiment()` drops a document whose task function raised, with no error
message; `write_experiment_jsonl` records each one as `status: "dropped"` so
document counts still add up.

The host and port come from a tracing config (`configs/tracing.example.yaml`);
the keys come from `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`, never a file.
"""

from __future__ import annotations

import concurrent.futures
import difflib
import logging
import subprocess
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1_tpu.core.config import load_config
from open_r1_tpu.evaluation import scoring
from open_r1_tpu.evaluation.config import _reject_unknown_keys
from open_r1_tpu.evaluation.generate import iter_documents, make_task, render_messages
from open_r1_tpu.evaluation.summary import jsonl_path, ok_record, write_jsonl
from open_r1_tpu.evaluation.taskpack import (
    dataset_name as taskpack_dataset_name,
)
from open_r1_tpu.evaluation.taskpack import (
    derive_task_spec,
    resolve_task_configs,
)

LOGGER = logging.getLogger(__name__)


# The complete key set each section accepts. A key outside this set is either
# a typo or a stale setting from a schema that moved on, exactly as
# `evaluation.run._reject_unknown_keys` treats an eval recipe.
LANGFUSE_KEYS = {"host", "port"}

SECTIONS = {"langfuse": LANGFUSE_KEYS}


def _require_port(field: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError(f"{field} must be a TCP port number")


def _require_nonempty_str(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def validate_tracing_config(config: dict[str, Any]) -> None:
    """Fail early for a tracing config mistake, before anything is launched."""
    for section in config:
        if section not in SECTIONS:
            close = difflib.get_close_matches(str(section), sorted(SECTIONS), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"Unknown configuration section {section!r}{hint}")

    for section, allowed in SECTIONS.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
        _reject_unknown_keys(section, config[section], allowed)

    langfuse = config["langfuse"]
    _require_nonempty_str("langfuse.host", langfuse.get("host"))
    _require_port("langfuse.port", langfuse.get("port"))


def load_tracing_config(
    path: str | Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load a tracing config and apply dotted `section.key=value` overrides."""
    return load_config(path, overrides, validator=validate_tracing_config)


def build_langfuse_client(tracing_config: Mapping[str, Any]) -> Any:
    """A `Langfuse` client from this project's own tracing config -- the
    `langfuse` section. Shared by `evaluation.dataset_sync` and
    `evaluation.experiment`.
    """
    from langfuse import Langfuse

    langfuse_section = tracing_config["langfuse"]
    base_url = f"http://{langfuse_section['host']}:{langfuse_section['port']}"
    return Langfuse(base_url=base_url)


LANGFUSE_FLUSH_TIMEOUT_SECS = 10.0


class LangfuseGuard:
    """Every Langfuse call funnelled through here, so a dead Langfuse costs a
    missing trace, score, or dataset item, never a generation. The first
    failure in a run logs a full warning; every subsequent one is counted
    silently, and the total is logged once at the end -- a dead Langfuse must
    not spam the log once per document across a 1,819-document tier.
    """

    def __init__(self, client: Any):
        self.client = client
        self.failures = 0
        self._warned = False

    def flush(self) -> None:
        # The SDK's own flush() has no timeout, and a hung export must not
        # hang the run -- so it is bounded from outside, in a worker thread
        # (safe here: unlike evaluation.scoring.compute_scores, nothing
        # Langfuse does depends on running on the main thread).
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self.client.flush).result(
                    timeout=LANGFUSE_FLUSH_TIMEOUT_SECS
                )
        except Exception:
            self.failures += 1
            LOGGER.warning("Langfuse flush failed or timed out", exc_info=True)

    def create_dataset(self, **kwargs: Any) -> Any | None:
        """Ensure one Langfuse dataset exists before any
        `create_dataset_item` call reaches it -- `create_dataset_item` 404s
        against a dataset that was never created, which is exactly what
        happened the first time `evaluation.dataset_sync` ran against a live
        Langfuse without this call. `POST /api/public/v2/datasets`'s
        generated client (checked against the installed `langfuse==4.14.5`)
        has no documented conflict response for an existing name -- every
        status this endpoint's spec models falls through to a 200, so a
        repeat call is expected to return the existing dataset rather than
        error. If that ever turns out wrong in practice, it would show up
        here as `failures` climbing on every routine re-sync, not as a
        silent 404 per item -- a far cheaper failure mode to notice.
        Returns the `Dataset`, or `None` if Langfuse failed.
        """
        try:
            return self.client.create_dataset(**kwargs)
        except Exception:
            self.failures += 1
            if not self._warned:
                LOGGER.warning(
                    "Langfuse call failed; continuing without ensuring "
                    "further datasets exist (further failures are counted, "
                    "not logged)",
                    exc_info=True,
                )
                self._warned = True
            return None

    def create_dataset_item(self, **kwargs: Any) -> Any | None:
        """Every `evaluation.dataset_sync` upsert funnelled through here: a
        dead Langfuse must cost a missing dataset item, never stop the sync
        -- and the run it gates -- from starting. Returns the created
        `DatasetItem`, or `None` if Langfuse failed.
        """
        try:
            return self.client.create_dataset_item(**kwargs)
        except Exception:
            self.failures += 1
            if not self._warned:
                LOGGER.warning(
                    "Langfuse call failed; continuing without syncing "
                    "further dataset items (further failures are counted, "
                    "not logged)",
                    exc_info=True,
                )
                self._warned = True
            return None


# Namespace for deterministic dataset item ids: the same (dataset, doc_id)
# always yields the same id, so create_dataset_item upserts rather than
# accumulating.
_ITEM_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "open-r1-tpu-dataset-item")


def item_id(dataset: str, doc_id: str) -> str:
    """Deterministic id for one dataset's one document. Stable across
    re-runs of `sync_task` against the same dataset name and document id,
    which is what makes a re-sync an upsert rather than a duplicate.
    """
    return str(uuid.uuid5(_ITEM_ID_NAMESPACE, f"{dataset}:{doc_id}"))


def ensure_dataset(guard: LangfuseGuard, name: str) -> bool:
    """Ensure Langfuse dataset `name` exists before any `create_dataset_item`
    call reaches it. Returns `True` once the dataset is known to exist
    (created now, or already there from an earlier sync); `False` if
    Langfuse could not be reached -- `LangfuseGuard.create_dataset` has
    already logged/counted that failure, and the caller must skip syncing
    this task's items rather than attempt them against a dataset that
    almost certainly does not exist.
    """
    return guard.create_dataset(name=name) is not None


def sync_task(
    guard: LangfuseGuard,
    task: str,
    config: Any,
    *,
    name: str,
    system_prompt: str | None,
    max_samples: int | None,
) -> int:
    """Upsert every document of one task's evaluation split as a Langfuse
    dataset item under dataset `name`.

    `name` is computed by the caller (`taskpack.dataset_name`, from a real
    `LightevalTaskConfig`'s derived `TaskSpec`) rather than here, so this
    function's own tests can stub `config` down to just what `build_doc`/
    `iter_documents` need, without deriving a task spec (which would need a
    real dataset load for its best-effort `example` field). Returns the item
    count.
    """
    documents = iter_documents(config, max_samples=max_samples)
    for doc_id, row in documents:
        doc = scoring.build_doc(config.prompt_function, row, task)
        guard.create_dataset_item(
            dataset_name=name,
            input=render_messages(doc, system_prompt),
            expected_output=scoring.single_gold(doc, task, doc_id),
            metadata={
                "task": task,
                "doc_id": doc_id,
                "specific": doc.specific,
                "query": doc.query,
            },
            id=item_id(name, doc_id),
        )
    return len(documents)


def sync_recipe(
    guard: LangfuseGuard, settings: Mapping[str, Any]
) -> dict[str, tuple[str, int]]:
    """Sync every task `settings["tasks"]` names. Returns
    `{task: (dataset_name, item_count)}`; a task whose dataset could not be
    ensured reports `item_count == 0` and is skipped, not retried per item.
    """
    task_names: Sequence[str] = list(settings["tasks"])
    resolved = resolve_task_configs(task_names)
    results: dict[str, tuple[str, int]] = {}
    for task in task_names:
        config = resolved[task]
        spec = derive_task_spec(task, config)
        name = taskpack_dataset_name(
            task, spec, max_samples=settings.get("max_samples")
        )
        if not ensure_dataset(guard, name):
            LOGGER.warning(
                "could not ensure dataset %s exists; skipping %s's documents "
                "rather than upserting each against a dataset that almost "
                "certainly doesn't exist",
                name,
                task,
            )
            results[task] = (name, 0)
            continue
        count = sync_task(
            guard,
            task,
            config,
            name=name,
            system_prompt=settings.get("system_prompt"),
            max_samples=settings.get("max_samples"),
        )
        LOGGER.info("synced %s -> dataset %s (%d item(s))", task, name, count)
        results[task] = (name, count)
    return results


def lighteval_evaluator(task_name: str) -> Callable[..., list[Any]]:
    """Factory for the per-document evaluator `dataset.run_experiment()`
    calls: `evaluator(*, input, output, expected_output, metadata, **kwargs)
    -> list[Evaluation]`.

    Closed over `task_name`'s live metric objects, resolved fresh through
    `evaluation.taskpack.resolve_task_configs` on every call -- never
    deserialized from the committed task pack; see that module's docstring
    for why. Calls `build_model_response`, `compute_scores`, `run_level_fields`,
    `coerce_fields`, in that order.

    `output` is the dict `evaluation.task_fn.make_task`'s task function
    returns (`{"text", "finish_reason", "completion_tokens", ...}`), not a
    bare completion string -- `"text"` is what gets scored, and
    `"finish_reason"`/`"completion_tokens"` feed `run_level_fields` exactly as
    the generation outcome did before. `evaluation.experiment` reads the same
    `output` dict again, independently, to persist its JSONL record, so
    nothing downstream ever has to read a fact back out of Langfuse.

    Always returns a list, so a `SampleLevelMetricGrouping` (e.g. `ifeval`'s
    four accuracies) produces several named scores rather than being
    flattened into one -- `coerce_fields` already drops the list-valued
    metrics (`inst_level_*_acc`) that have no single-document scalar meaning;
    see `coerce_score`. A `ScoringResult` with `failed_metrics` adds one more:
    `Evaluation(name="scoring_failed", value=1.0, metadata={"failed_metrics":
    ..., "errors": ...})` -- visible in Langfuse rather than only in a log
    line, and readable back locally (its `metadata`) by `evaluation.experiment`
    without another Langfuse round trip.
    """
    from langfuse import Evaluation

    from open_r1_tpu.evaluation.taskpack import resolve_task_configs

    metrics = list(resolve_task_configs([task_name])[task_name].metrics)

    def evaluator(
        *, output: Any, expected_output: Any, metadata: Mapping[str, Any], **kwargs: Any
    ) -> list[Any]:
        text = output["text"] if isinstance(output, Mapping) else output
        doc = scoring.doc_from_item(expected_output, metadata, task_name)
        model_response = scoring.build_model_response(text)
        result = scoring.compute_scores(doc, model_response, metrics)

        fields = dict(result.scores)
        if isinstance(output, Mapping):
            fields.update(
                scoring.run_level_fields(
                    completion_tokens=output.get("completion_tokens"),
                    finish_reason=str(output.get("finish_reason", "")),
                )
            )

        evaluations = [
            Evaluation(name=name, value=value, data_type=data_type)
            for name, (value, data_type) in scoring.coerce_fields(fields).items()
        ]
        if result.failed_metrics:
            evaluations.append(
                Evaluation(
                    name="scoring_failed",
                    value=1.0,
                    data_type="NUMERIC",
                    metadata={
                        "failed_metrics": list(result.failed_metrics),
                        "errors": dict(result.errors),
                    },
                )
            )
        return evaluations

    return evaluator


# Evaluations `lighteval_evaluator` posts that are not a LightEval metric
# name -- run-level facts and the failure marker -- excluded from a JSONL
# record's `scores` dict, keeping `result.scores` (LightEval's own metric
# names) separate from `evaluation.scoring.run_level_fields`.
_RUN_LEVEL_EVALUATION_NAMES = frozenset(
    {"completion_tokens", "truncated", "scoring_failed"}
)


def _git_commit() -> str | None:
    """Best-effort provenance: `None` outside a git checkout or without
    `git` on `PATH`, never a hard failure -- this is metadata, not a
    correctness input.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _record_from_item_result(
    item_result: Any, *, task: str, seed: int
) -> dict[str, Any]:
    """One JSONL record built from an `ExperimentItemResult`."""
    output = item_result.output if isinstance(item_result.output, Mapping) else {}
    evaluations_by_name = {
        evaluation.name: evaluation for evaluation in item_result.evaluations
    }
    scores = {
        name: evaluation.value
        for name, evaluation in evaluations_by_name.items()
        if name not in _RUN_LEVEL_EVALUATION_NAMES
    }

    scoring_failed = evaluations_by_name.get("scoring_failed")
    failed_metrics: list[str] = []
    scoring_errors: dict[str, str] = {}
    if scoring_failed is not None and scoring_failed.metadata:
        failed_metrics = list(scoring_failed.metadata.get("failed_metrics", []))
        scoring_errors = dict(scoring_failed.metadata.get("errors", {}))

    item = item_result.item
    return ok_record(
        doc_id=item.metadata["doc_id"],
        task=task,
        seed=seed,
        gold=item.expected_output,
        query=item.metadata.get("query"),
        output=output,
        scores=scores,
        failed_metrics=failed_metrics,
        scoring_errors=scoring_errors,
        trace_id=item_result.trace_id,
    )


def _dropped_record(item: Any, *, task: str, seed: int) -> dict[str, Any]:
    """A document `run_experiment` returned no result for at all -- see the
    module docstring's note on why no error message survives to here.
    """
    return {
        "status": "dropped",
        "doc_id": item.metadata["doc_id"],
        "task": task,
        "seed": seed,
        "error": (
            "run_experiment did not return a result for this item; see "
            "Langfuse's own log for the raised exception"
        ),
    }


def write_experiment_jsonl(
    result: Any, dataset: Any, *, task: str, seed: int, output_path: Path
) -> None:
    """Persist one `(seed, task)`'s `ExperimentResult` as the JSONL
    `evaluation.reduce.reduce_seed` reads -- one line per dataset item,
    whether `run_experiment` returned a result for it or dropped it.
    """
    seen_ids = {item_result.item.id for item_result in result.item_results}
    records = [
        _record_from_item_result(item_result, task=task, seed=seed)
        for item_result in result.item_results
    ]
    records += [
        _dropped_record(item, task=task, seed=seed)
        for item in dataset.items
        if item.id not in seen_ids
    ]
    write_jsonl(output_path, records)


def run_experiment_for_task_seed(
    langfuse_client: Any,
    *,
    task: str,
    seed: int,
    dataset_name: str,
    settings: Mapping[str, Any],
    client: Any,
    output_dir: Path,
    run_metadata: Mapping[str, Any],
) -> Any:
    """Run and persist one `(task, seed)`'s experiment against the synced
    Langfuse dataset `dataset_name`.
    """
    dataset = langfuse_client.get_dataset(dataset_name)
    fingerprint = dataset_name.rsplit("@", 1)[-1]
    result = dataset.run_experiment(
        name=f"{settings['served_model_name']}-{settings['tier']}-seed{seed}",
        task=make_task(settings, client=client),
        evaluators=[lighteval_evaluator(task)],
        max_concurrency=int(settings["max_concurrency"]),
        metadata={**run_metadata, "dataset_fingerprint": fingerprint},
    )

    output_path = jsonl_path(output_dir, seed, task)
    write_experiment_jsonl(
        result, dataset, task=task, seed=seed, output_path=output_path
    )
    LOGGER.info(
        "seed %d %s -> %s (%d/%d item(s) scored)",
        seed,
        task,
        output_path,
        len(result.item_results),
        len(dataset.items),
    )
    return result


def sync_datasets(langfuse_client: Any, settings: Mapping[str, Any]) -> dict[str, str]:
    """Sync the recipe's tasks into Langfuse and return `{task: dataset}`.

    Idempotent, so it runs before every traced evaluation rather than as a
    separate step. Stops on any Langfuse failure: an experiment against an
    incomplete dataset would score fewer documents than the recipe asks for.
    """
    guard = LangfuseGuard(langfuse_client)
    synced = sync_recipe(guard, settings)
    guard.flush()
    if guard.failures:
        raise RuntimeError(
            f"{guard.failures} Langfuse call(s) failed while syncing the "
            "recipe's datasets; fix Langfuse and re-run, or drop "
            "--tracing-config to evaluate without it"
        )
    return {task: name for task, (name, _) in synced.items()}


def run_langfuse(
    settings: Mapping[str, Any],
    *,
    langfuse_client: Any,
    client: Any,
    output_dir: Path,
    recipe_path: str | None,
) -> None:
    datasets = sync_datasets(langfuse_client, settings)
    run_metadata = {
        "recipe_path": recipe_path,
        "git_commit": _git_commit(),
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_new_tokens": settings["max_new_tokens"],
    }
    for seed in settings["seeds"]:
        for task in settings["tasks"]:
            run_experiment_for_task_seed(
                langfuse_client,
                task=task,
                seed=seed,
                dataset_name=datasets[task],
                settings=settings,
                client=client,
                output_dir=output_dir,
                run_metadata=run_metadata,
            )
