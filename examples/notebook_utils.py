"""Helpers shared by the tutorial notebooks in this directory.

The notebooks never import JAX themselves: training and sampling run as
separate commands through `run`, so the TPU is free again as soon as each one
ends. `read_scalars` and `plot_scalars` read back the metrics a run wrote.
"""

from __future__ import annotations

import codecs
import os
import shlex
import signal
import struct
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    """The open-r1-tpu checkout that contains `start` (default: the working
    directory)."""
    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        if (directory / "pyproject.toml").is_file() and (
            directory / "src" / "open_r1_tpu"
        ).is_dir():
            return directory
    raise FileNotFoundError(f"{here} is not inside an open-r1-tpu checkout")


def count_tpu_chips() -> int:
    """Count the TPU chips JAX sees, from a separate process so that this one
    never holds the TPU."""
    code = "import jax; print(jax.devices()); print(jax.device_count())"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            "JAX could not start. Is another process using the TPU?\n"
            + result.stderr[-2000:]
        )
    devices, count = result.stdout.strip().splitlines()[-2:]
    print(devices)
    if "Tpu" not in devices:
        raise RuntimeError("JAX sees no TPU: run this notebook on a TPU VM")
    return int(count)


def run(*command: object) -> None:
    """Run a command, show its output as it arrives, and raise if it fails.

    Interrupting the cell interrupts the command, which is how a training run
    should be stopped: it lets the process release the TPU.
    """
    args = [str(part) for part in command]
    print("$", shlex.join(args), flush=True)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    assert process.stdout is not None
    try:
        while chunk := os.read(process.stdout.fileno(), 4096):
            sys.stdout.write(decoder.decode(chunk))
            sys.stdout.flush()
    except KeyboardInterrupt:
        process.send_signal(signal.SIGINT)
        process.wait()
        raise
    if process.wait() != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}")


def read_scalars(log_dir: str | Path) -> pd.DataFrame:
    """Every scalar a run logged under `log_dir`: one row per tag and step.

    Tunix writes TensorBoard event files. Each is a sequence of records: an
    8-byte length, a 4-byte checksum, the event itself and another checksum.
    A step logged more than once keeps its last value.
    """
    from tensorboardX.proto import event_pb2

    rows = []
    for path in sorted(Path(log_dir).rglob("events.out.tfevents.*")):
        data = path.read_bytes()
        offset = 0
        while offset + 12 <= len(data):
            (length,) = struct.unpack_from("<Q", data, offset)
            start, end = offset + 12, offset + 12 + length
            if end + 4 > len(data):
                break  # the run is still writing this record
            event = event_pb2.Event.FromString(data[start:end])
            offset = end + 4
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    rows.append(
                        {
                            "tag": value.tag,
                            "step": int(event.step),
                            "value": float(value.simple_value),
                            "time": float(event.wall_time),
                        }
                    )
    frame = pd.DataFrame(rows, columns=["tag", "step", "value", "time"])
    return (
        frame.sort_values("time")
        .drop_duplicates(["tag", "step"], keep="last")
        .sort_values(["tag", "step"])
        .reset_index(drop=True)
    )


def plot_scalars(
    scalars: pd.DataFrame, tags: Sequence[str], *, title: str = "", ax=None
):
    """Plot the named tags against the step; a missing tag is reported, not
    plotted, along with the tags that do exist."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    missing = []
    for tag in tags:
        series = scalars[scalars["tag"] == tag]
        if series.empty:
            missing.append(tag)
            continue
        ax.plot(series["step"], series["value"], marker=".", label=tag)
    if missing:
        print(f"Not logged: {', '.join(missing)}")
        print(f"Logged tags: {', '.join(sorted(scalars['tag'].unique()))}")
    ax.set_xlabel("step")
    ax.set_title(title)
    if len(tags) > len(missing):
        ax.legend()
    return ax
