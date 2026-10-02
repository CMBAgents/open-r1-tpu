# Tutorials

Two notebooks that train a small model end to end on one TPU v6e chip,
explaining each step on the way. They assume no previous experience of
fine-tuning or reinforcement learning.

| Notebook | What you do |
| --- | --- |
| [`01_sft.ipynb`](01_sft.ipynb) | Supervised fine-tuning: turn GSM8K's worked solutions into chat conversations, see which tokens the model learns from and how packing works, then train Qwen2.5-0.5B to reason in `<think>` tags and box its answer |
| [`02_grpo.ipynb`](02_grpo.ipynb) | Reinforcement learning with GRPO: try the reward functions, work out group-relative advantages by hand, then train the SFT model on its own sampled answers and compare all three models |

Run them in order: the second starts from the model the first trains.

The other files support them:

- `recipes/`: the two training recipes and the system prompt both use. Their
  comments explain each setting.
- `answer_questions.py`: asks a model the GSM8K test questions on the TPU and
  scores its replies; it also runs on its own.
- `notebook_utils.py`: runs commands from a notebook and reads back the
  metrics a run logged.

## Open the notebooks

The notebooks run on the TPU VM, in the environment
`./scripts/setup_tpu_vm.sh` builds (see the [README](../README.md#install)).
Jupyter is not part of that environment; `uv run --with` adds it for one
session without changing it. From the repository root on the VM:

```bash
uv run --frozen --with jupyterlab --with ipywidgets --with matplotlib \
  jupyter lab --no-browser --port 8888
```

Then forward the port from your own computer and open the
`http://localhost:8888/lab?token=...` address Jupyter printed:

```bash
gcloud compute tpus tpu-vm ssh YOUR_VM --zone YOUR_ZONE -- -L 8888:localhost:8888
```

Add `--tunnel-through-iap` if the VM has no external IP address. Start Jupyter
under `tmux` if you want training to survive a dropped SSH connection.

In VS Code connected to the VM over Remote-SSH, open a notebook and choose
`.venv/bin/python` as its kernel, after running
`uv pip install ipykernel ipywidgets matplotlib` with `.venv` active. A later
`uv sync` removes those three again.

## What they write

Everything the notebooks download or produce goes to directories git ignores:
the base model to `models/Qwen2.5-0.5B`, the data to `data/gsm8k-think`, and
checkpoints, logs, the trained models and the scored answers to `artifacts/`.
Training resumes from the newest checkpoint it finds, so delete a run's
directory under `artifacts/` to start it again from scratch.

The notebooks never import JAX themselves: training and sampling run as
separate commands, so the TPU is free whenever no command is running. Stop a
run by interrupting its cell.
