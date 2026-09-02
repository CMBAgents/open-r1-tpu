# Local Langfuse (self-hosted)

Self-hosted Langfuse (web, worker, Postgres, ClickHouse, Redis, MinIO). This
is where `open_r1_tpu.evaluation.experiment` drives an evaluation from --
`dataset.run_experiment()` reads the dataset `evaluation.dataset_sync`
populated here, generates and scores each document, and posts the resulting
trace and scores itself. This file covers only operating the stack; see
`open_r1_tpu.evaluation.experiment`/`.dataset_sync`/`.task_fn` for the
pipeline itself.

## Setup

```bash
cp docker/langfuse/.env.example docker/langfuse/.env
# edit docker/langfuse/.env: passwords, keys, and (if the defaults collide
# with something else on the host) ports.
```

The `LANGFUSE_WEB_PORT` in `.env` and the tracing config's
(`configs/tracing.yaml`) `langfuse.host`/`langfuse.port` describe the same
endpoint. They must agree, or `evaluation.experiment`/`.dataset_sync` will
connect to the wrong instance.

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
MinIO's volumes are disposable.

On a freshly provisioned VM, rebuild the instance and its datasets in two
commands:

```bash
scripts/run_langfuse_stack.sh up
# once healthy (headless init has created the org/project/user/keys), sync
# every task the recipe you intend to run needs:
python -m open_r1_tpu.evaluation.dataset_sync \
  --config recipes/<model>/eval/<tier>.yaml --tracing-config configs/tracing.yaml
```

`dataset_sync`'s own deterministic item ids (`uuid5(dataset_name, doc_id)`)
mean a re-run against an unchanged task upserts every item and creates
nothing new, so this is safe to run again on an already-populated instance.
Traces and scores themselves are not recovered this way -- they are written
fresh by the next `scripts/run_eval_tpu.sh` launch, since nothing durable
outside this stack captures them.

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

## The manual UI check

Nothing here automates looking at the result: after a tier-0 smoke run,
open the UI (above), find the experiment `evaluation.experiment` just
created for that `(task, seed)`, and confirm a trace is there with its
input, output, and scores attached. That is a human confirming a web page
renders what it should -- do it once after standing up a new Langfuse
instance, and again after any change to `evaluation.experiment`/`.task_fn`/
`.scoring`.

## Python environment

`langfuse` is part of the frozen evaluation environment (`pyproject.toml`'s
`eval` extra) -- it is on the critical path for `open_r1_tpu.evaluation.experiment`,
which talks to it in-process, so it lives beside `openai`, `lighteval`, and
the rest of that extra. `uv sync --extra eval` (or
`scripts/setup_tpu_vm.sh --with-eval`) is the whole setup, on the VM and on a
Mac alike.
