# Install

Everything runs on a Google Cloud TPU VM. Run every command from the
repository root on the VM.

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
| `--with-eval` | Also installs LightEval and builds the vLLM image, for [evaluation](evaluation.md) |
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

    The shipped recipes log to W&B. To run without it, pass
    `training.wandb.enabled=false` to training or
    `reporting.wandb.enabled=false` to an evaluation.

## What is where

| Path | Contents |
| --- | --- |
| [`src/open_r1_tpu/sft/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/src/open_r1_tpu/sft) | SFT data preparation, training and preflight |
| [`src/open_r1_tpu/grpo/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/src/open_r1_tpu/grpo) | GRPO prompt loading, rewards and training |
| [`src/open_r1_tpu/evaluation/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/src/open_r1_tpu/evaluation) | Evaluation: recipe config, vLLM server, generation, scoring, summary, optional Langfuse tracing |
| [`src/open_r1_tpu/model/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/src/open_r1_tpu/model) | Model and tokenizer loading, LoRA, optimizer, metrics, export, checkpoint restore |
| [`src/open_r1_tpu/core/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/src/open_r1_tpu/core) | Recipe loading, the shared command line, logging |
| [`recipes/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/recipes) | Training and evaluation recipes; see [Recipes](recipes.md) |
| [`scripts/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/scripts) | Launchers and tools: VM setup, evaluation, the vLLM container, chat, export |
| [`docker/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/docker) | The pinned vLLM TPU image, and the optional Langfuse stack |
| [`configs/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/configs) | The template for the optional Langfuse tracing config |
| [`examples/`](https://github.com/CMBAgents/open-r1-tpu/tree/main/examples) | [Tutorial notebooks](tutorials.md) for SFT and GRPO on one chip, with their recipes |

Staged models go in `models/`, datasets in `data/`, and checkpoints, logs and
exports in `artifacts/`. Git ignores all three.

## Next steps

- New to fine-tuning or reinforcement learning: the [tutorials](tutorials.md).
- Ready for a full run: [distillation](sft.md) or [GRPO](grpo.md).
