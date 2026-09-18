#!/usr/bin/env python3
"""Aggregate immutable E029 shard metrics without concealing replay seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_execution(shard_roots: Iterable[Path], output_dir: Path, *, expected_conditions: int) -> dict[str, Any]:
    """Combine complete shard metric files and retain every seed-specific row."""
    roots = list(shard_roots)
    if not roots:
        raise ValueError("at least one E029 shard root is required")
    if expected_conditions <= 0:
        raise ValueError("expected_conditions must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable E029 aggregation: {output_dir}")

    required = {"family", "modality", "k_in", "replay_seed", "hit_at_10", "ndcg_at_10", "mrr_at_10"}
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    for root in roots:
        metrics_path = root / "materialized" / "exact_metrics.csv"
        receipt_path = root / "materialized" / "materialization_receipt.json"
        if not metrics_path.is_file() or not receipt_path.is_file():
            raise ValueError(f"incomplete E029 materialized shard: {root}")
        frame = pd.read_csv(metrics_path)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"E029 shard metrics lack columns {sorted(missing)}: {root}")
        frame["replay_seed"] = frame["replay_seed"].astype("Int64")
        frames.append(frame)
        sources.append({"root": str(root), "metrics_sha256": sha256(metrics_path), "materialization_receipt_sha256": sha256(receipt_path)})

    per_seed = pd.concat(frames, ignore_index=True)
    key_columns = ["family", "modality", "k_in", "replay_seed"]
    if per_seed.duplicated(key_columns).any():
        raise ValueError("duplicate E029 condition across materialized shards")
    if len(per_seed) != expected_conditions:
        raise ValueError(f"expected {expected_conditions} E029 conditions, found {len(per_seed)}")
    per_seed.sort_values(key_columns, inplace=True, kind="stable")

    grouping = ["family", "modality", "k_in"]
    numeric = [column for column in ("hit_at_10", "ndcg_at_10", "mrr_at_10", "recall_at_10", "exact_any_gold_at_10") if column in per_seed]
    seed_mean = per_seed.groupby(grouping, as_index=False, dropna=False)[numeric].mean()
    seed_mean["seed_count"] = per_seed.groupby(grouping, dropna=False).size().to_numpy()
    seed_mean.sort_values(grouping, inplace=True, kind="stable")

    output_dir.mkdir(parents=True)
    per_seed_path = output_dir / "per_seed_exact_metrics.csv"
    seed_mean_path = output_dir / "seed_mean_exact_metrics.csv"
    per_seed.to_csv(per_seed_path, index=False)
    seed_mean.to_csv(seed_mean_path, index=False)
    receipt = {
        "schema_version": "b2-e029-execution-aggregation.v1",
        "conditions": len(per_seed),
        "sources": sources,
        "per_seed_exact_metrics.csv": sha256(per_seed_path),
        "seed_mean_exact_metrics.csv": sha256(seed_mean_path),
    }
    (output_dir / "aggregation_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-conditions", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = aggregate_execution(args.shard_root, args.output_dir, expected_conditions=args.expected_conditions)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
