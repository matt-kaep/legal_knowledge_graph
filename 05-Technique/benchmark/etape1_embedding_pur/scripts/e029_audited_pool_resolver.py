"""Resolve a reranking pool from the location and checksum sealed by its audit."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Mapping


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_audited_pool(
    audit_job_files: Iterable[Mapping[str, str]],
    *,
    data_root: Path,
    family: str,
    modality: str,
    k_in: int,
) -> Path:
    """Return the unique hash-validated audited pool under ``data_root``."""
    filename = f"{family}_{modality}_kin{k_in}.jsonl"
    matches = [row for row in audit_job_files if Path(row["path"]).name == filename]
    if len(matches) != 1:
        raise ValueError(f"expected one audited pool named {filename}, found {len(matches)}")

    declared_path = Path(matches[0]["path"])
    try:
        relative_path = declared_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"audited pool lies outside data root: {declared_path}") from exc

    resolved_path = data_root / relative_path
    expected_sha256 = matches[0]["sha256"]
    if not resolved_path.is_file() or sha256(resolved_path) != expected_sha256:
        raise ValueError(f"unverified audited pool: {resolved_path}")
    return resolved_path
