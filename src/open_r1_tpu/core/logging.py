"""Stderr logging shared by training and evaluation workflows.

Tunix, Orbax and JAX all log through absl, which routes every one of them
through a single logger named "absl" at INFO. Neither the level nor the logger
name can tell a training step from a library's internal bookkeeping, and the
loudest bookkeeping in this stack sits on a hot path: PeftTrainer calls
CheckpointManager.save on every optimizer step and lets Orbax's save policy
decide whether to write, so Orbax rebuilds its handler registry and logs six
INFO lines around each step's single loss line whether or not anything is
saved.

Records from the packages in NOISY_PACKAGES are demoted to DEBUG rather than
dropped, so --log-level debug brings them back and nothing is lost. Only INFO
is demoted; warnings and errors keep their level whatever their source. To
quieten another package that hides behind absl, add its import name to
NOISY_PACKAGES.
"""

from __future__ import annotations

import logging
import os

LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
}

# Packages whose INFO output is per-step bookkeeping rather than progress.
NOISY_PACKAGES = ("orbax",)

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class _DemoteNoisyPackages(logging.Filter):
    """Demote INFO records originating in `packages` to DEBUG.

    Attach this to a logger, not to a handler. Logger.callHandlers compares a
    record's level against each handler before running that handler's filters,
    so a demotion made there would relabel the record without suppressing it.
    """

    def __init__(self, packages: tuple[str, ...]) -> None:
        super().__init__()
        # Match a path component so that, say, tunix/sft/orbax_utils.py is not
        # mistaken for Orbax.
        self._paths = tuple(f"{os.sep}{name}{os.sep}" for name in packages)

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno != logging.INFO:
            return True
        if any(path in record.pathname for path in self._paths):
            record.levelno = logging.DEBUG
            record.levelname = logging.getLevelName(logging.DEBUG)
        return True


def configure_logging(
    level: int = logging.INFO,
    packages: tuple[str, ...] = NOISY_PACKAGES,
) -> None:
    """Log to stderr at `level`, with `packages` demoted to DEBUG."""
    # Imported here rather than at module scope for the side effect of its
    # ordering: absl builds its ABSLLogger through logging.getLogger("absl"),
    # so if anything registers a plain logger under that name first, absl keeps
    # using the standard library's findCaller and every record then reports
    # absl's own file instead of the calling module's, which is all the filter
    # above has to go on. Importing it late also keeps this module importable
    # without absl installed.
    from absl import logging as absl_logging

    handler = logging.StreamHandler()
    # basicConfig sets the level on the root logger and leaves handlers at
    # NOTSET. A demoted record is suppressed by a handler level, so the handler
    # needs one of its own.
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])
    absl_logging.get_absl_logger().addFilter(_DemoteNoisyPackages(packages))
