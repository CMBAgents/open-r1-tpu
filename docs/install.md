# Install

This page installs the software on a TPU VM; [Setting up on
TPU](tpu-setup.md) creates the VM. Run every command from the repository root
on the VM.

## Set up the VM

```bash
git clone https://github.com/CMBAgents/open-r1-tpu.git
cd open-r1-tpu
./scripts/setup_tpu_vm.sh            # add --with-eval for the evaluation stack
```

The script installs `uv`, the CPython pinned in `.python-version`, and the
locked environment in `.venv`, then checks JAX sees the TPU and runs the unit
tests. It is safe to re-run.

| Flag | Effect |
| --- | --- |
| `--with-eval` | Also installs LightEval and builds the vLLM image, for [benchmarking](benchmarking.md). Use it on the evaluation VM. |
| `--skip-verify` | Skips the checks, when another job holds the TPU |

To install by hand instead, see
[Training in depth](training.md#environment).

## Configure

The script also writes `~/.open-r1-tpu.env`, kept out of the repository, with
the settings to fill in commented out. Set the ones you need:

```bash
export HF_TOKEN=hf_...                  # Hub downloads
export WANDB_ENTITY=your-user-or-team   # W&B
export WANDB_PROJECT=your-project
```

Then load it with the environment, in every new shell:

```bash
source ~/.open-r1-tpu.env
source .venv/bin/activate
wandb login
```

!!! note "Running without Weights & Biases"

    The shipped recipes log training curves to
    [Weights & Biases](https://wandb.ai). To run without it, pass
    `training.wandb.enabled=false` to training or
    `reporting.wandb.enabled=false` to an evaluation.

## Where things go

Staged models go in `models/`, datasets in `data/`, and checkpoints, logs and
finished models in `artifacts/`. Git ignores all three; copy anything worth
keeping to a [bucket](tpu-setup.md#buckets).

## Next steps

- New to fine-tuning or reinforcement learning: the [tutorials](tutorials.md).
- Ready for a full run: [download a base model](base-models.md), then
  [SFT](sft.md) or [RL](grpo.md).
