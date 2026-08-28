# Local Langfuse (self-hosted)

Self-hosted Langfuse (web, worker, Postgres, ClickHouse, Redis, MinIO), for
viewing evaluation traces captured by the trace proxy (`docker/trace-proxy/`)
and scored by `open_r1_tpu.tracing.scores`. See the observability plan in the
tracking repo for the full architecture; this file covers only operating this
stack.

## Setup

```bash
cp docker/langfuse/.env.example docker/langfuse/.env
# edit docker/langfuse/.env: passwords, keys, and (if the defaults collide
# with something else on the host) ports.
```

The `LANGFUSE_WEB_PORT` in `.env` and the tracing config's
(`configs/tracing.yaml`) `langfuse.host`/`langfuse.port` describe the same
endpoint. They must agree, or the score pass and the ingester will authenticate
against the wrong instance.

## Start, stop, inspect

```bash
scripts/run_langfuse_stack.sh up
scripts/run_langfuse_stack.sh ps
scripts/run_langfuse_stack.sh logs langfuse-web
scripts/run_langfuse_stack.sh down
```

Everything binds to `127.0.0.1` only. It is not reachable from outside the
host it runs on by design (see "Viewing the UI" below).

## Ephemeral by design

The TPU VM this stack normally runs on is a flex-start instance that
auto-deletes. Nothing here is backed up: Postgres, ClickHouse, Redis, and
MinIO's volumes are disposable. The GCS bucket the trace proxy writes to is
the durable record.

On a freshly provisioned VM, rebuild the instance in two commands:

```bash
scripts/run_langfuse_stack.sh up
# once healthy (headless init has created the org/project/user/keys):
python -m open_r1_tpu.tracing.ingest --config configs/tracing.yaml --once
```

The ingester's own idempotency (deterministic trace ids, upsert-by-id) means
replaying the entire bucket backlog against a brand-new Langfuse reproduces
every trace exactly once, whatever ran before the VM was deleted.

**Datasets recover the same way, through a different command.** Under the
Langfuse-native runner (`open_r1_tpu.evaluation.runner`/`.experiment`,
`eval-langfuse-native-plan.md`), the datasets `dataset.run_experiment()`
drives are not in the bucket at all -- they never went through the trace
proxy. Rebuilding them after a fresh instance is one more idempotent command,
using whichever recipe named the tasks that mattered:

```bash
python -m open_r1_tpu.evaluation.dataset_sync \
  --config recipes/<model>/eval/<tier>.yaml --tracing-config configs/tracing.yaml
```

`open_r1_tpu.evaluation.dataset_sync`'s own deterministic item ids
(`uuid5(dataset_name, doc_id)`) mean a re-run against an unchanged task
upserts every item and creates nothing new -- the same property the
ingester's trace ids give the command above, just for datasets instead of
traces. Once `eval-langfuse-native-plan.md`'s Task 6 lands (the trace
proxy/ingester deletion, gated on its own tier-1 parity gate), this becomes
the *only* rebuild story for this stack; until then, a run still using the
older litellm-proxy capture path needs both commands.

## Viewing the UI

Run from a workstation, with the placeholders below filled in for the real
project, zone, and VM name:

```bash
gcloud compute tpus tpu-vm ssh <vm-name> \
  --project <project> --zone <zone> \
  -- -L <port>:127.0.0.1:<port>
```

`<port>` is `LANGFUSE_WEB_PORT` from `docker/langfuse/.env` on the VM (default
`3000`). With the tunnel open, browse to `http://localhost:<port>` and sign in
with `LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`.

## Resource note

This stack (web, worker, Postgres, ClickHouse, Redis, MinIO) shares the host
with the vLLM container during an evaluation run. The host has ample CPU and
RAM for both, but any run whose *throughput* is being measured -- the
generation speed benchmark -- should be launched with this stack stopped
(`scripts/run_langfuse_stack.sh down`), since it is one more set of processes
competing for the same CPU cores and disk I/O that a speed number is trying to
isolate.

## Local rehearsal and the manual UI check

`tests/test_trace_capture_rehearsal.py` exercises the real wire path end to
end -- a stub OpenAI-compatible server standing in for vLLM, the real trace
proxy container, the real `gcs_logger.py` callback, and the real ingester --
against a scratch GCS prefix, asserting against a faked Langfuse client
rather than a running one:

```bash
TRACE_REHEARSAL_GCS_BUCKET=<a bucket you can freely write to and delete from> \
  uv run pytest tests/test_trace_capture_rehearsal.py -m integration
```

It needs Docker and that bucket; it skips cleanly without either. It does not
need this Langfuse stack, since it fakes the client -- proving the trace
lands correctly is independent of Langfuse actually rendering it.

**Treat a passing rehearsal as a hard gate before the first `TRACE_PROXY=1`
launch on the VM** (and after any change to the proxy container's config,
image pin, or `gcs_logger.py`). It is the only check that exercises what the
unit tests cannot see -- the container's networking, the dependencies
actually present inside the pinned image, and real GCS auth -- which is
precisely where a silent capture failure would live. On the VM itself the
rehearsal is cheap: Docker and ADC are already there, so it is one pytest
command against a scratch prefix before hours of eval traffic depend on the
same path.

The one thing nothing here automates is looking at the result: after running
either this rehearsal or the real ingester against a Langfuse instance that
*is* up, open the UI (above) and confirm the trace you just created is
actually there, with its messages, completion, and (after
`open_r1_tpu.tracing.scores` runs) its scores attached. That last check is
manual by nature -- it is a human confirming a web page renders what it
should -- do it once after standing up a new Langfuse instance, and again
after any change to `open_r1_tpu.tracing.ingest`/`.scores` or
`docker/trace-proxy/gcs_logger.py`.

## Python environment

`langfuse` is part of the frozen evaluation environment (`pyproject.toml`'s
`eval` extra) -- it is on the critical path for `open_r1_tpu.evaluation.runner`,
the Langfuse-native runner that talks to it in-process, so it lives beside
`openai`, `lighteval`, and the rest of that extra rather than in a separate
one. There is no more `.venv-tracing`: `uv sync --extra eval` (or
`scripts/setup_tpu_vm.sh --with-eval`) is the whole setup, on the VM and on a
Mac alike, for the runner and for `open_r1_tpu.tracing.ingest`/`.scores`
(the older litellm-proxy capture path these superseded, still present until
it is deleted).
