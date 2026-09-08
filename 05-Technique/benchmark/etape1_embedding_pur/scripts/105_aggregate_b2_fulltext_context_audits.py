#!/usr/bin/env python3
"""Aggregate immutable E029 context-audit receipts into one complete matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


SHARD_TO_AGGREGATE_SCHEMA = {
    "b2-e029-fulltext-context-audit.v1": "b2-e029-fulltext-context-audit-aggregate.v1",
    "b2-e029-context-audit.v2": "b2-e029-context-audit-aggregate.v2",
}
CONTRACT_KEYS = (
    "model",
    "context_limit_tokens",
    "max_output_tokens",
    "candidate_text_policy",
    "prompt_files",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"context audit is not a JSON object: {path}")
    if value.get("schema_version") not in SHARD_TO_AGGREGATE_SCHEMA:
        raise ValueError(f"unexpected shard schema_version in {path}: {value.get('schema_version')!r}")
    if not isinstance(value.get("conditions"), list) or not value["conditions"]:
        raise ValueError(f"context audit has no conditions: {path}")
    return value


def _condition_key(condition: Mapping[str, Any]) -> tuple[str, str, int, str | None]:
    try:
        replay_seed = None if condition.get("replay_seed") is None else str(condition["replay_seed"])
        return str(condition["family"]), str(condition["modality"]), int(condition["k_in"]), replay_seed
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid context-audit condition: {condition!r}") from error


def aggregate_reports(report_paths: Iterable[Path], *, expected_conditions: int) -> dict[str, object]:
    """Combine complete shard receipts while rejecting any contract drift or gap."""
    paths = list(report_paths)
    if not paths:
        raise ValueError("at least one context-audit receipt is required")
    if expected_conditions <= 0:
        raise ValueError("expected_conditions must be positive")

    contract: dict[str, Any] | None = None
    shard_schema_version: str | None = None
    source_reports: list[dict[str, str]] = []
    job_files: list[dict[str, Any]] = []
    conditions: list[dict[str, Any]] = []
    seen_conditions: set[tuple[str, str, int, str | None]] = set()

    for path in paths:
        payload = _read_report(path)
        observed_schema = str(payload["schema_version"])
        if shard_schema_version is None:
            shard_schema_version = observed_schema
        elif observed_schema != shard_schema_version:
            raise ValueError(f"context-audit schema mismatch: {path}")
        observed_contract = {key: payload.get(key) for key in CONTRACT_KEYS}
        if contract is None:
            contract = observed_contract
        else:
            for key in CONTRACT_KEYS:
                if observed_contract[key] != contract[key]:
                    raise ValueError(f"context-audit contract mismatch for {key}: {path}")
        source_reports.append({"path": str(path), "sha256": sha256(path)})
        raw_job_files = payload.get("job_files")
        if not isinstance(raw_job_files, list):
            raise ValueError(f"context audit has no job_files list: {path}")
        job_files.extend(raw_job_files)
        for raw_condition in payload["conditions"]:
            if not isinstance(raw_condition, dict):
                raise ValueError(f"context-audit condition is not an object: {path}")
            key = _condition_key(raw_condition)
            if key in seen_conditions:
                raise ValueError(f"duplicate condition across context-audit receipts: {key}")
            seen_conditions.add(key)
            conditions.append(raw_condition)

    if len(conditions) != expected_conditions:
        raise ValueError(f"expected {expected_conditions} conditions, found {len(conditions)}")
    assert contract is not None and shard_schema_version is not None
    conditions.sort(key=lambda row: tuple("" if value is None else value for value in _condition_key(row)))
    compatible_conditions = sum(bool(row.get("compatible")) for row in conditions)
    overflow_questions = sum(len(row.get("overflow_qids", [])) for row in conditions)
    return {
        "schema_version": SHARD_TO_AGGREGATE_SCHEMA[shard_schema_version],
        **contract,
        "source_reports": source_reports,
        "job_files": job_files,
        "conditions": conditions,
        "expected_conditions": expected_conditions,
        "compatible_conditions": compatible_conditions,
        "incompatible_conditions": expected_conditions - compatible_conditions,
        "overflow_questions": overflow_questions,
    }


def write_immutable_report(path: Path, report: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate context audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True, help="Immutable shard receipt; repeat for each shard.")
    parser.add_argument("--expected-conditions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = aggregate_reports(args.report, expected_conditions=args.expected_conditions)
    write_immutable_report(args.output, report)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256(args.output),
        "conditions": len(report["conditions"]),
        "compatible_conditions": report["compatible_conditions"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
