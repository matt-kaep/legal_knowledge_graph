"""Portable checkout and data-root resolution for benchmark tooling."""

from __future__ import annotations

import os
from pathlib import Path


def _discover_checkout(anchor: Path) -> Path:
    candidate = anchor.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"Could not discover a Git checkout from {anchor}")


def resolve_repo_root(anchor: Path | None = None) -> Path:
    """Return the code checkout from ``LKG_REPO`` or the current Git tree."""

    configured = os.environ.get("LKG_REPO")
    if configured:
        return Path(configured).expanduser().resolve()
    return _discover_checkout(anchor or Path(__file__))


def resolve_data_root(anchor: Path | None = None) -> Path:
    """Return the data checkout, defaulting to the code checkout."""

    configured = os.environ.get("LKG_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return resolve_repo_root(anchor)
