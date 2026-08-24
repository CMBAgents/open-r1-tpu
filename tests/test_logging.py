import logging

from open_r1_tpu.core.logging import (
    LOG_LEVELS,
    NOISY_PACKAGES,
    _DemoteNoisyPackages,
)

SITE = "/venv/lib/python3.13/site-packages"
ORBAX = f"{SITE}/orbax/checkpoint/_src/handlers/pytree_handler.py"
TUNIX = f"{SITE}/tunix/sft/peft_trainer.py"


def _record(pathname: str, level: int = logging.INFO) -> logging.LogRecord:
    # absl logs every library through one logger and records the calling
    # module's own file, which is all that separates them.
    return logging.LogRecord(
        name="absl",
        level=level,
        pathname=pathname,
        lineno=1,
        msg="message",
        args=(),
        exc_info=None,
    )


class _RecordingHandler(logging.Handler):
    def __init__(self, level: int) -> None:
        super().__init__(level)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _logger_at(level: int, name: str) -> tuple[logging.Logger, _RecordingHandler]:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.filters.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addFilter(_DemoteNoisyPackages(NOISY_PACKAGES))
    handler = _RecordingHandler(level)
    logger.addHandler(handler)
    return logger, handler


def test_noisy_package_info_is_demoted_rather_than_dropped():
    record = _record(ORBAX)

    assert _DemoteNoisyPackages(NOISY_PACKAGES).filter(record) is True
    assert record.levelno == logging.DEBUG
    assert record.levelname == "DEBUG"


def test_noisy_package_problems_keep_their_level():
    for level in (logging.WARNING, logging.ERROR):
        record = _record(ORBAX, level)

        assert _DemoteNoisyPackages(NOISY_PACKAGES).filter(record) is True
        assert record.levelno == level


def test_training_progress_keeps_its_level():
    record = _record(TUNIX)

    assert _DemoteNoisyPackages(NOISY_PACKAGES).filter(record) is True
    assert record.levelno == logging.INFO


def test_only_whole_path_components_are_matched():
    # A module named after Orbax is not Orbax.
    record = _record(f"{SITE}/tunix/sft/orbax_utils.py")

    assert _DemoteNoisyPackages(NOISY_PACKAGES).filter(record) is True
    assert record.levelno == logging.INFO


def test_any_package_can_be_quietened():
    record = _record(f"{SITE}/fsspec/caching.py")

    assert _DemoteNoisyPackages(("fsspec",)).filter(record) is True
    assert record.levelno == logging.DEBUG


def test_demotion_is_what_an_info_handler_suppresses():
    # The filter always returns True, so suppression comes from the handler
    # level. This is why it belongs on a logger: callHandlers compares the
    # level before a handler's own filters would run.
    logger, handler = _logger_at(logging.INFO, "test_noise_at_info")

    logger.handle(_record(ORBAX))
    logger.handle(_record(TUNIX))

    assert [record.pathname for record in handler.records] == [TUNIX]


def test_debug_brings_the_demoted_records_back():
    logger, handler = _logger_at(logging.DEBUG, "test_noise_at_debug")

    logger.handle(_record(ORBAX))
    logger.handle(_record(TUNIX))

    assert [record.pathname for record in handler.records] == [ORBAX, TUNIX]


def test_log_levels_cover_the_command_line_choices():
    assert LOG_LEVELS["debug"] < LOG_LEVELS["info"] < LOG_LEVELS["warning"]
