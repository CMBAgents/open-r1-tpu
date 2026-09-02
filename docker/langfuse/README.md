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
scripts/gen_langfuse_env.sh
```

This writes `docker/langfuse/.env` and `configs/tracing.yaml` together, with a
fresh set of secrets and every value that must match already matching:
`DATABASE_URL`'s embedded password, the `LANGFUSE_S3_*` MinIO credentials, and
`langfuse.port` against `LANGFUSE_WEB_PORT`. `.env` has no variable
interpolation, so those are the pairs a hand-edit of `.env.example` gets wrong.
Both files are gitignored; re-running is refused without `--force` so a live
stack's secrets are not rotated out from under its volumes.

That is the single-host setup, where both files land on the same machine and
both point at loopback. If the evaluation runs somewhere else, see "Running
the stack on its own host" below instead.

To hand-edit instead, `cp docker/langfuse/.env.example docker/langfuse/.env` and
fill in every value, keeping the inline "must equal" notes. Either way,
`.env`'s `LANGFUSE_WEB_BIND`/`LANGFUSE_WEB_PORT` and the tracing config's
`langfuse.host`/`langfuse.port` describe the same endpoint -- the first pair as
the server publishes it, the second as the client reaches it -- and must agree,
or `evaluation.experiment`/`.dataset_sync` will connect to the wrong instance.

## Running the stack on its own host

The stack and the evaluation do not have to share a machine, and there are
good reasons to separate them: the evaluation host may be a flex-start TPU VM
that auto-deletes, taking every trace with it, and the six containers here
otherwise compete for the CPU and disk the generation benchmark is trying to
measure. Split that way, each host gets one of the two files, and the endpoint
is the only thing they must agree on.

On the host running the stack -- publish langfuse-web on an interface the
evaluation host can reach, instead of the loopback default:

```bash
scripts/gen_langfuse_env.sh --web-bind <stack host address> --no-tracing-config
scripts/run_langfuse_stack.sh up
scripts/gen_langfuse_env.sh --print-keys      # copy the two lines across
```

On the host running the evaluation -- the client half only, no secrets and no
`.env`:

```bash
scripts/gen_langfuse_env.sh --tracing-only --langfuse-host <stack host address>
```

then append the two `export` lines from `--print-keys` to the file the eval
launch sources (e.g. `~/.tpu-env`) on *this* host, and launch as usual with
`TRACE_CONFIG=configs/tracing.yaml`.

Three things this does not do for you:

- **The firewall.** `--web-bind` only decides which interface Docker publishes
  on; nothing here controls who may connect. Admit that port from the
  evaluation host alone. Note that on a cloud VPC a broad
  "allow all internal traffic" rule may already permit it from every host on
  the network -- adding a narrower rule does not revoke a broader one.
- **The datastores.** Postgres, ClickHouse, Redis and MinIO stay bound to
  loopback in every deployment. Only langfuse-web's bind address is
  configurable, and it is the only port anything outside the host needs.
- **Encryption.** The client speaks plain HTTP to `langfuse.host`. That is
  fine inside a trusted private network and nowhere else; a stack reachable
  beyond one needs a TLS terminator in front of it.

## Start, stop, inspect

```bash
scripts/run_langfuse_stack.sh up
scripts/run_langfuse_stack.sh ps
scripts/run_langfuse_stack.sh logs langfuse-web
scripts/run_langfuse_stack.sh down
```

Everything binds to `127.0.0.1` by default, so out of the box the stack is
not reachable from outside the host it runs on (see "Viewing the UI" below).
`LANGFUSE_WEB_BIND` is the one exception, and only for langfuse-web -- see
"Running the stack on its own host".

## Durability is a property of the host, not of this stack

Nothing here is backed up: the Postgres, ClickHouse, Redis and MinIO volumes
hold the only copy of every trace and score, and the trace proxy that once
mirrored them to GCS is gone. So how long the data lives is decided entirely
by which host you put it on. On a flex-start TPU VM that auto-deletes, it is
gone with the VM -- which is the strongest argument for the separate host
above. On a long-lived VM it persists, and rebuilding is an exception rather
than the routine.

Either way, on a freshly provisioned host, rebuild the instance and its
datasets:

```bash
scripts/gen_langfuse_env.sh                 # .env + configs/tracing.yaml are gone with the old host
scripts/run_langfuse_stack.sh up
scripts/gen_langfuse_env.sh --print-keys >> ~/.tpu-env   # the file the eval launch sources
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

The UI is always reached by SSH port-forward, never by exposing it publicly.
Run from a workstation, with the placeholders filled in for the real project,
zone, and VM name (`gcloud compute ssh` for an ordinary VM,
`gcloud compute tpus tpu-vm ssh` for a TPU one):

```bash
gcloud compute ssh <vm-name> \
  --project <project> --zone <zone> \
  -- -L <port>:<bind>:<port>
```

`<port>` is `LANGFUSE_WEB_PORT` from `docker/langfuse/.env` on that VM (default
`3000`) and `<bind>` is its `LANGFUSE_WEB_BIND` (default `127.0.0.1`). The
forward target must be the bind address: sshd opens the connection from the VM
itself, so once the stack publishes on a routable interface rather than
loopback, `127.0.0.1` there has nothing listening on it. Note this is the VM
running the *stack*, which in a two-host deployment is not the VM running the
evaluation.

With the tunnel open, browse to `http://localhost:<port>` and sign in with
`LANGFUSE_INIT_USER_EMAIL` / `LANGFUSE_INIT_USER_PASSWORD`. The browser origin
stays `localhost` whatever the bind address is, which is why `NEXTAUTH_URL`
does too.

## Resource note

On a single-host deployment this stack (web, worker, Postgres, ClickHouse,
Redis, MinIO) shares the host with the vLLM container during an evaluation
run. The host has ample CPU and RAM for both, but any run whose *throughput*
is being measured -- the generation speed benchmark -- should be launched with
this stack stopped (`scripts/run_langfuse_stack.sh down`), since it is one
more set of processes competing for the same CPU cores and disk I/O that a
speed number is trying to isolate.

Moving the stack to its own host removes that contention outright, which is
the other reason to do it. The generation client still shares the evaluation
process, so a speed benchmark is not perfectly isolated either way -- but six
database containers are no longer in the picture.

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
