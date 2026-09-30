"""Stderr logging shared by training and evaluation workflows.

Tunix, Orbax and JAX all log through absl at INFO, and Orbax logs several lines
on every optimizer step. INFO records from NOISY_PACKAGES are demoted to DEBUG
rather than dropped, so --log-level debug brings them back; warnings and
errors keep their level.
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
    # Records name the calling module's file, which is all the filter above
    # goes on, only if absl registered its "absl" logger before anything else
    # created a plain one. Imported here so this module loads without absl.
    from absl import logging as absl_logging

    handler = logging.StreamHandler()
    # A demoted record is suppressed by a handler level, which basicConfig
    # leaves at NOTSET.
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(_FORMAT))
    logging.basicConfig(level=level, handlers=[handler])
    absl_logging.get_absl_logger().addFilter(_DemoteNoisyPackages(packages))
