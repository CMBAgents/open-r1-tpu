"""Installed-package queries for preflight checks and run records."""

from __future__ import annotations

from importlib import metadata


def installed_version(distribution: str) -> str:
    """The installed version of `distribution`, or "unknown" if it is absent."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"
