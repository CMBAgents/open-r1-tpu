# Setting up on TPU

The pipeline runs in a Google Cloud project. This page creates what it needs:

| What | Type | Used for |
| --- | --- | --- |
| A bucket | Cloud Storage | Base models, datasets, checkpoints and finished models |
| A training VM | TPU v6e-4 | [SFT](sft.md) and [RL](grpo.md) |
| An evaluation VM | TPU v6e-1 | [Benchmarking](benchmarking.md), [chat](chat.md) and the [tutorials](tutorials.md) |
| A Langfuse VM (optional) | Ordinary CPU VM | Evaluation traces; see [Langfuse](langfuse.md) |

Replace the capitalised placeholders (`YOUR_PROJECT`, `YOUR_ZONE` and so on)
with your own values. Once a VM is up, [Install](install.md) sets up the
software on it.

## Before you start

You need a Google Cloud project with billing enabled and TPU quota, and the
[gcloud CLI](https://cloud.google.com/sdk/docs/install) on your own computer.
Point gcloud at the project and turn on the TPU API:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT
gcloud services enable tpu.googleapis.com
```

TPU quota is granted per region. Check what the project has, and which TPU
types a zone offers:

```bash
gcloud compute regions describe YOUR_REGION \
  --format="table(quotas.metric, quotas.limit, quotas.usage)" | grep -i tpu

gcloud alpha compute tpus accelerator-types list --zone=YOUR_ZONE \
  --format="value(acceleratorType)" | grep v6e
```

## TPUs in brief

A *TPU VM* is an ordinary Linux machine with TPU chips attached. This project
uses TPU v6e (Trillium), which has 32 GB of high-bandwidth memory (HBM) per
chip. The number after the dash is the number of chips: a v6e-1 has one, a
v6e-4 has four on the same host.

A model too large for one chip's memory is split across several. The recipe's
`model.mesh` says how: the 1.5B SFT recipe's `[fsdp, tp] = [2, 2]` shards the
weights and the batch two ways (`fsdp`) and splits each layer's matrices two
ways (`tp`, tensor parallelism), using all four chips of a v6e-4.
JAX compiles the training step for that layout with XLA the first time it
runs, which is why the first step is much slower than the rest.

!!! warning "One process per TPU"

    Only one process can use a VM's TPU chips at a time, so training,
    benchmarking and chat on the same VM run one after another. This is why
    benchmarking gets its own [small VM](#the-small-evaluation-vm).

## Buckets

A Cloud Storage bucket keeps everything that must outlive a VM: staged base
models and datasets, checkpoints, and finished models. Create it in the same
region as your TPU VMs, which keeps copies fast and avoids transfer charges:

```bash
gcloud storage buckets create gs://YOUR_BUCKET \
  --location=YOUR_REGION --default-storage-class=STANDARD
```

A layout that mirrors the repository's own directories keeps copies simple:

```text
gs://YOUR_BUCKET/models/       # base models, as models/ on the VM
gs://YOUR_BUCKET/data/         # datasets, as data/
gs://YOUR_BUCKET/artifacts/    # checkpoints and finished models, as artifacts/
```

TPU VMs act as the project's Compute Engine default service account. Give it
read and write access to the bucket:

```bash
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT --format="value(projectNumber)")
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

On the VM, copy what a run needs to local disk before it starts. Tunix's
Hugging Face loader contacts the Hub even for weights already on disk, so the
recipes load from local directories (`model_source: local`) instead. A
re-run of `rsync` resumes an interrupted copy:

```bash
gcloud storage rsync --recursive gs://YOUR_BUCKET/models/Qwen2.5-Math-1.5B \
  models/Qwen2.5-Math-1.5B
```

Training can write its checkpoints straight to the bucket by setting
`training.checkpoint_dir` to a `gs://` path. A finished model is always
written to local disk first; copy it up afterwards:

```bash
gcloud storage rsync --recursive artifacts/YOUR_RUN/merged \
  gs://YOUR_BUCKET/artifacts/YOUR_RUN/merged
```

## TPU VMs

### Choosing a size

| VM | Chips | Runs |
| --- | --- | --- |
| v6e-4 | 4 | The full SFT and RL recipes. The 1.5B SFT recipe peaks at 19.7 GiB per chip with a 26k-token window. |
| v6e-1 | 1 | Benchmarking, chat, and the tutorials' 0.5B model |

### Choosing how to pay

TPU v6e VMs are requested as *queued resources*: you ask for a TPU, and the
request waits until Google Cloud has one free. There are two ways to pay:

- **On demand.** The VM runs until you delete it, at the full price.
- **Flex-start.** Discounted, but the VM runs for at most
  `--max-run-duration` (up to seven days) and is then deleted with its disk.
  Keep everything you need in the bucket.

### Creating a VM

```bash
gcloud alpha compute tpus queued-resources create YOUR_VM \
  --zone=YOUR_ZONE \
  --accelerator-type=v6e-4 \
  --runtime-version=v2-alpha-tpuv6e \
  --node-id=YOUR_VM \
  --provisioning-model=flex-start \
  --max-run-duration=72h \
  --valid-until-duration=72h
```

Drop `--provisioning-model` and `--max-run-duration` for an on-demand VM.

| Flag | Meaning |
| --- | --- |
| `--accelerator-type` | The TPU type and chip count, such as `v6e-1` or `v6e-4` |
| `--runtime-version` | The VM image; `v2-alpha-tpuv6e` for v6e |
| `--max-run-duration` | Flex-start only: how long the VM runs once it starts, up to `168h` |
| `--valid-until-duration` | How long the request waits in the queue before giving up |

!!! note "Projects without external IP addresses"

    If an organisation policy forbids external IP addresses
    (`compute.vmExternalIpAccess`), add `--internal-ips`, turn on Private
    Google Access for the region's subnet so the VM can still reach Google
    services, and connect with `--tunnel-through-iap`:

    ```bash
    gcloud compute networks subnets update default \
      --region=YOUR_REGION --enable-private-ip-google-access
    ```

### Waiting for it

```bash
gcloud alpha compute tpus queued-resources describe YOUR_VM --zone=YOUR_ZONE
gcloud alpha compute tpus queued-resources list --zone=-    # every zone
```

A request moves through three states:

| State | Meaning |
| --- | --- |
| `WAITING_FOR_RESOURCES` | Queued until a TPU is free |
| `PROVISIONING` | A TPU has been allocated and the VM is booting |
| `ACTIVE` | The VM is up and ready for SSH |

!!! tip "Asking in several zones at once"

    When capacity is short, submit the same request in several zones that
    offer v6e and keep whichever starts first. Delete the others while they
    are still `WAITING_FOR_RESOURCES`: a request cannot be deleted while it is
    `PROVISIONING`.

For a flex-start VM, this prints when it will be deleted:

```bash
gcloud alpha compute tpus queued-resources describe YOUR_VM --zone=YOUR_ZONE \
  --format="value(tpu.nodeSpec[0].node.schedulingConfig.terminationTimestamp)"
```

### Connecting

```bash
gcloud alpha compute tpus tpu-vm ssh YOUR_VM --zone=YOUR_ZONE
```

Add `--tunnel-through-iap` if the VM has no external IP address. The first
connection to a new VM copies your SSH key to it and can take half a minute.
To reach a web page served on the VM, such as Jupyter, forward its port:

```bash
gcloud alpha compute tpus tpu-vm ssh YOUR_VM --zone=YOUR_ZONE -- -L 8888:localhost:8888
```

Run long jobs under `tmux` on the VM, so they survive a dropped connection.

### Deleting it

An on-demand VM is billed until it is deleted. Copy anything you need to the
bucket first:

```bash
gcloud alpha compute tpus queued-resources delete YOUR_VM --zone=YOUR_ZONE --quiet
```

## The small evaluation VM

Benchmarking serves a model with vLLM on a single chip, and training holds
every chip of its VM for hours. On one VM, scoring a checkpoint would mean
stopping training. A second, one-chip v6e-1 avoids that:

1. The training VM finishes a run, or a checkpoint is
   [exported](training.md#checkpoints-and-export), and the model is copied to
   the bucket.
2. The evaluation VM copies it down and benchmarks it, while the training VM
   moves on to the next run.

Create it as above with `--accelerator-type=v6e-1`, in the same region as the
bucket, and install with `./scripts/setup_tpu_vm.sh --with-eval`, which also
builds the vLLM image. To benchmark on a multi-chip VM instead, see
[Running evaluations](evaluation.md#limits) for pinning the server to one
chip.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Constraint compute.vmExternalIpAccess violated` | The organisation forbids external IP addresses | Add `--internal-ips` when creating the VM |
| SSH to a `10.x.x.x` address times out | The VM has only an internal address | Add `--tunnel-through-iap` |
| `NOT_FOUND` for the VM on `ssh` or `describe` | The VM is still being created | Wait for the request to reach `ACTIVE` |
| Deleting a request fails | It is `PROVISIONING` | Wait until it is `ACTIVE`, then delete |
| `PermissionDenied` from Cloud Storage | The VM's service account has no access to the bucket | Grant it `roles/storage.objectAdmin`, as [above](#buckets) |
| `git clone` asks for a username on a public repository, with `expected flush after ref listing` | Git's HTTP/2 transport fails on some private networks | `git config --global http.version HTTP/1.1` |
| `START_SESSION failed` on a multi-chip VM after a run crashed | The crashed run left the chips' interconnect session held | Launch with `LIBTPU_INIT_ARGS=--noenable_tpunetd_client`, or re-create the VM |
| A stopped run still holds the TPU | Only its tmux session was killed | Interrupt or kill the Python process itself |
