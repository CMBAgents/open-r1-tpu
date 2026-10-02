# Langfuse

[Langfuse](https://langfuse.com) is an open-source platform for observing
language-model applications. It records *traces*: what a model was asked,
what it answered, how long it took, and the scores attached to the answer.
Its web interface lets you search, read and compare them.

## Why use it for benchmarking

A summary says a model scored 54.7% on MATH-500. It does not say why it got
the other 45.3% wrong. Langfuse holds every reply, so you can read them: did
the model reason badly, run out of tokens, forget to box its answer, or give
a right answer the scorer missed?

With Langfuse connected, an evaluation:

- syncs each benchmark into a Langfuse **dataset**, one item per problem;
- runs each benchmark and seed as an **experiment** over that dataset;
- records each problem's prompt, reply and **scores** as a trace.

Two experiments on the same dataset, say before and after a change to
training, can then be compared problem by problem.

It is optional. Without it, evaluations write the same results files and
summary; Langfuse adds a way to read them. Datasets are named after the
benchmark and a fingerprint of its prompt and scoring, so a changed task never
mixes with an old one.

## How it runs

Langfuse is self-hosted from `docker/langfuse/`: six containers (the web
interface, a background worker, and the Postgres, ClickHouse, Redis and MinIO
stores behind them), managed with `scripts/run_langfuse_stack.sh`. Nothing is
sent to an outside service.

It can share the evaluation VM, or run on a VM of its own. A separate,
ordinary CPU VM (an `e2-standard-4` is plenty) keeps the traces when a TPU VM
is deleted, which matters for flex-start VMs, and keeps six containers from
competing with vLLM for the CPU.

## Setting it up

On one VM, from the repository root:

```bash
scripts/gen_langfuse_env.sh                 # writes the stack's secrets and the client config
scripts/run_langfuse_stack.sh up
scripts/gen_langfuse_env.sh --print-keys >> ~/.open-r1-tpu.env
source ~/.open-r1-tpu.env
```

Then pass the client config to an evaluation:

```bash
TRACE_CONFIG=configs/tracing.yaml \
  RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
```

On a VM of its own, publish the web interface on an address the evaluation
VM can reach, and point the evaluation VM at it:

```bash
# On the Langfuse VM
scripts/gen_langfuse_env.sh --web-bind LANGFUSE_VM_ADDRESS --no-tracing-config
scripts/run_langfuse_stack.sh up
scripts/gen_langfuse_env.sh --print-keys     # copy the two lines it prints

# On the evaluation VM
scripts/gen_langfuse_env.sh --tracing-only --langfuse-host LANGFUSE_VM_ADDRESS
# then add the two copied lines to ~/.open-r1-tpu.env and source it
```

!!! warning "Firewall and encryption"

    Allow the web port (3000 by default) from the evaluation VM only. The
    client speaks plain HTTP, which is fine inside a private network but
    nowhere wider.

## Viewing the traces

The web interface is never exposed publicly. Reach it through an SSH tunnel
from your own computer:

```bash
gcloud compute ssh YOUR_LANGFUSE_VM --zone=YOUR_ZONE -- -L 3000:BIND_ADDRESS:3000
```

`BIND_ADDRESS` is `127.0.0.1` when Langfuse shares the VM, or the address
passed to `--web-bind`. Use `gcloud compute tpus tpu-vm ssh` when Langfuse
runs on a TPU VM. Then open `http://localhost:3000` and sign in with
`LANGFUSE_INIT_USER_EMAIL` and `LANGFUSE_INIT_USER_PASSWORD` from
`docker/langfuse/.env` on the Langfuse VM.

## Keeping the traces

Nothing is backed up: the traces live in Docker volumes on the Langfuse VM
and go when it does. The results files and summary on the evaluation VM are
the lasting record. A new Langfuse VM gets the datasets back on its own, the
first time each benchmark is traced.

The [Langfuse README](https://github.com/CMBAgents/open-r1-tpu/blob/main/docker/langfuse/README.md)
has the full operating detail.
