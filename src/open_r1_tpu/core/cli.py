"""The command line every recipe-driven entry point shares.

--config RECIPE [--log-level LEVEL] [section.key=value ...]
"""

from __future__ import annotations

import argparse

from open_r1_tpu.core.logging import LOG_LEVELS, configure_logging


def recipe_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="YAML recipe path")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=sorted(LOG_LEVELS),
        type=str.lower,
        help="Stderr log level. debug restores the demoted library logs.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dotted recipe overrides such as training.max_steps=4",
    )
    return parser


def parse_recipe_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse the command line and configure logging at the chosen level."""
    args = parser.parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
    return args
