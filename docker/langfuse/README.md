# Self-hosted Langfuse

Self-hosted Langfuse (web, worker, Postgres, ClickHouse, Redis, MinIO) for
tracing evaluations. It is optional: without it, `scripts/run_eval_tpu.sh` runs
locally and writes the same results. With `TRACE_CONFIG` set, each run syncs
the recipe's tasks into Langfuse datasets, then `dataset.run_experiment()`
generates and scores each document and posts its trace and scores here. This
file covers operating the stack; see `open_r1_tpu.evaluation.run` for the
pipeline.

## Setup

```bash
scripts/gen_langfuse_env.sh
scripts/run_langfuse_stack.sh up
scripts/gen_langfuse_env.sh --print-keys >> ~/.open-r1-tpu.env
source ~/.open-r1-tpu.env
TRACE_CONFIG=configs/tracing.yaml \
  RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

The first command writes two gitignored files: `docker/langfuse/.env`, the
stack's settings, filled in from `.env.example` with fresh secrets, and
`configs/tracing.yaml`, where the evaluation's client connects. Every value
that must match another already does: `DATABASE_URL`'s embedded password, the
`LANGFUSE_S3_*` MinIO credentials, and `langfuse.port` against
`LANGFUSE_WEB_PORT`. Re-running is refused without `--force`, so a running
stack's secrets are not rotated out from under its volumes.

To write `.env` by hand instead, copy `.env.example` and replace every
`changeme`, keeping the pairs its comments mark as equal. Either way, `.env`'s
`LANGFUSE_WEB_BIND`/`LANGFUSE_WEB_PORT` and the tracing config's
`langfuse.host`/`langfuse.port` name the same endpoint, from the server's and
the client's side.

## Running the stack on its own host

The stack and the evaluation need not share a machine. A separate host keeps
the traces when the evaluation host is deleted, and keeps six containers from
competing with a throughput measurement. Each host then gets one of the two
files.

On the host running the stack, publish langfuse-web on an address the
evaluation host can reach:

```bash
scripts/gen_langfuse_env.sh --web-bind <stack host address> --no-tracing-config
scripts/run_langfuse_stack.sh up
scripts/gen_langfuse_env.sh --print-keys      # copy the two lines across
```

On the host running the evaluation, write the client config only:

```bash
scripts/gen_langfuse_env.sh --tracing-only --langfuse-host <stack host address>
```

then append the two `export` lines from `--print-keys` to
`~/.open-r1-tpu.env` on this host, source it, and launch as usual with
`TRACE_CONFIG=configs/tracing.yaml`.

Three things this does not do for you:

- **The firewall.** `--web-bind` only decides which interface Docker publishes
  on. Admit that port from the evaluation host alone; on a cloud VPC, a broad
  "allow internal traffic" rule may already admit every host on the network,
  and a narrower rule does not revoke it.
- **The datastores.** Postgres, ClickHouse, Redis and MinIO stay bound to
  loopback in every deployment. Only langfuse-web's address is configurable.
- **Encryption.** The client speaks plain HTTP to `langfuse.host`, which is fine
  inside a trusted private network only; anything wider needs a TLS terminator
  in front.

## Start, stop, inspect

```bash
scripts/run_langfuse_stack.sh up
scripts/run_langfuse_stack.sh ps
scripts/run_langfuse_stack.sh logs langfuse-web
scripts/run_langfuse_stack.sh down
```

## Durability

Nothing here is backed up: the volumes hold the only copy of every trace and
score, so they live exactly as long as the host does. To rebuild on a new host,
run the setup commands above (`.env` and `configs/tracing.yaml` go with the old
host). Datasets come back on their own: every traced run syncs its recipe's
tasks first, and the deterministic item ids (`uuid5(dataset_name, doc_id)`)
make that an upsert. Old traces and scores are not recovered; the results JSONL
and summary on the evaluation host are the durable record.

## Viewing the UI

The UI is reached by SSH port-forward, never exposed publicly. From a
workstation, with the placeholders filled in (`gcloud compute ssh` for an
ordinary VM, `gcloud compute tpus tpu-vm ssh` for a TPU one):

```bash
gcloud compute ssh <vm-name> \
  --project <project> --zone <zone> \
  -- -L <port>:<bind>:<port>
```

`<port>` is `LANGFUSE_WEB_PORT` from the stack host's `docker/langfuse/.env`
(default `3000`) and `<bind>` its `LANGFUSE_WEB_BIND` (default `127.0.0.1`).
The target must be the bind address because sshd connects from the VM itself.

With the tunnel open, browse to `http://localhost:<port>` and sign in with
`LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`. The browser origin
is `localhost` whatever the bind address, which is why `NEXTAUTH_URL` is too.

After standing up a new instance, or changing `evaluation.run`, `.generate`,
`.traced` or `.scoring`, run the tier-0 smoke tier and check in the UI that the
experiment for each `(task, seed)` has traces with their input, output and
scores.

## Throughput measurements

On a single host this stack shares CPU and disk with the vLLM container. Stop
it (`scripts/run_langfuse_stack.sh down`), or run it on its own host, before
the generation speed benchmark.

## Python environment

The `langfuse` client is part of the `eval` extra (`uv sync --extra eval`, or
`scripts/setup_tpu_vm.sh --with-eval`) and is used only when `TRACE_CONFIG` is
set.
