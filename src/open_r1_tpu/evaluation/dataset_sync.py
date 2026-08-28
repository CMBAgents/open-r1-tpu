"""Sync one recipe's tasks into Langfuse datasets.

Task 1 of `eval-langfuse-native-plan.md`: builds the Langfuse datasets that
`evaluation.experiment`'s `dataset.run_experiment()` drives. One dataset per
task, one item per document -- the same documents `evaluation.runner` used to
generate against directly (`taskpack.resolve_task_configs`,
`runner.iter_documents`, `runner.render_messages`), so a document's identity
and prompt are identical whichever half of the pipeline produced them.

**`expected_output` is LightEval's gold string, not the bare answer.**
`math_500`'s prompt function sets `choices=["ANSWER: {solution}"]` with
`gold_index=0`, and its metric extracts over the *gold* as well as the
prediction -- tidying the `ANSWER:` prefix away would change what is compared.
`doc.get_golds()` is stored verbatim; see `evaluation.scoring.doc_from_item`,
which rebuilds an equivalent `Doc` from exactly this item at scoring time,
never by calling the prompt function again.

**A document's `query`/`specific` are captured once, here, and never
recomputed.** Most of this project's prompt functions are pure, but `gpqa`'s
is not: it shuffles its four answer choices with `random.randint` on every
call, so calling the prompt function a second time at scoring time would
judge a different shuffle than the one actually sent to the model. Storing
`doc.query` and `doc.specific` in the item's metadata and rebuilding the
scoring `Doc` from them (`doc_from_item`) rather than re-rendering is what
keeps every task correct, not just the deterministic ones.

**Dataset naming is `{task}@{fingerprint}`** (`taskpack.dataset_name`), so a
change to what is asked or how it is judged (dataset coordinates, revision,
prompt function, metrics) yields a new dataset rather than silently mixing
incomparable runs together. Item ids are `uuid5(dataset_name, doc_id)`, so
re-running this module against an unchanged task upserts every item and
creates nothing new -- this is also the recovery path for the ephemeral VM
Langfuse stack; see `docker/langfuse/README.md`.

Every Langfuse call is funnelled through `runner.LangfuseGuard`, so a dead
Langfuse costs a missing dataset item, never a crashed sync -- consistent
with `evaluation.runner`'s own use of the guard for traces and scores.

**The dataset itself must exist before any item is upserted into it.**
`create_dataset_item` does not create its dataset as a side effect -- against
a live Langfuse, every item 404s until `create_dataset` has been called for
that name at least once. `sync_recipe` calls `ensure_dataset` once per task,
before `sync_task`'s per-document loop, and skips that task's items entirely
if it fails, rather than attempting (and failing) every one of them
individually against a dataset known not to exist.

Run from the repository root::

    python -m open_r1_tpu.evaluation.dataset_sync \\
      --config recipes/Qwen3-1.7B-Math/eval/tier1_core.yaml \\
      --tracing-config configs/tracing.yaml
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from open_r1_tpu.evaluation import scoring
from open_r1_tpu.evaluation.runner import LangfuseGuard, iter_documents, render_messages
from open_r1_tpu.evaluation.taskpack import (
    dataset_name as taskpack_dataset_name,
)
from open_r1_tpu.evaluation.taskpack import (
    derive_task_spec,
    resolve_task_configs,
)

LOGGER = logging.getLogger(__name__)

# Namespace for deterministic dataset item ids: the same (dataset, doc_id)
# always yields the same id, so create_dataset_item upserts rather than
# accumulating -- matching runner._trace_id/tracing.scores._score_id's choice
# on the trace/score side.
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
        messages = render_messages(doc, system_prompt)
        golds = doc.get_golds()
        if len(golds) != 1:
            raise ValueError(
                f"{task} document {doc_id}: dataset_sync only supports a "
                "single gold per document -- evaluation.scoring.doc_from_item "
                f"rebuilds a single-choice Doc at scoring time -- got "
                f"{len(golds)} golds"
            )
        guard.create_dataset_item(
            dataset_name=name,
            input=messages,
            expected_output=golds[0],
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
        name = taskpack_dataset_name(task, spec)
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


def _parse_args() -> Any:
    import argparse

    from open_r1_tpu.core.logging import LOG_LEVELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML evaluation recipe")
    parser.add_argument(
        "--tracing-config", required=True, help="YAML tracing config (Langfuse section)"
    )
    parser.add_argument("--log-level", default="info", choices=sorted(LOG_LEVELS))
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    from open_r1_tpu.core.logging import LOG_LEVELS, configure_logging
    from open_r1_tpu.evaluation.run import load_eval_config, resolve_settings
    from open_r1_tpu.tracing.config import build_langfuse_client, load_tracing_config

    args = _parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
    settings = resolve_settings(load_eval_config(args.config, args.overrides))
    tracing_config = load_tracing_config(args.tracing_config)

    guard = LangfuseGuard(build_langfuse_client(tracing_config))
    results = sync_recipe(guard, settings)
    guard.flush()

    for task, (name, count) in sorted(results.items()):
        LOGGER.info("%s: dataset %s has %d item(s)", task, name, count)

    if guard.failures:
        # Unlike evaluation.runner (which only ever warns: generation is the
        # expensive part, and a trace is best-effort), a sync that could not
        # reach Langfuse is worth stopping on -- it is cheap, idempotent, and
        # nothing downstream should start against a dataset known to be
        # incomplete.
        raise SystemExit(
            f"dataset_sync: {guard.failures} Langfuse call(s) failed; the "
            "dataset(s) above are incomplete. Fix Langfuse and re-run -- "
            "this is idempotent."
        )


if __name__ == "__main__":
    main()
