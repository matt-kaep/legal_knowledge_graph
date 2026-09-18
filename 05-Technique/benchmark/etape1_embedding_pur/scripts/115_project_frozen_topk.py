#!/usr/bin/env python3
"""Derive an immutable top-K ranking projection without rerunning retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_top_k(ranking: pd.DataFrame, *, k: int, condition_columns: Iterable[str]) -> pd.DataFrame:
    """Select ranks 1..K while preserving every explicit ranking condition."""
    if k <= 0:
        raise ValueError("K must be positive")
    conditions = list(condition_columns)
    if len(conditions) != len(set(conditions)):
        raise ValueError("condition columns must be unique")
    required = {"qid", "rank", "item_id", *conditions}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"ranking lacks required columns: {sorted(missing)}")
    if "qid" in conditions or "rank" in conditions or "item_id" in conditions:
        raise ValueError("condition columns must not repeat qid, rank or item_id")

    projected = ranking.copy()
    projected["qid"] = projected["qid"].astype(str)
    projected["item_id"] = projected["item_id"].astype(str)
    numeric_rank = pd.to_numeric(projected["rank"], errors="raise")
    if (numeric_rank % 1 != 0).any():
        raise ValueError("ranking rank values must be integers")
    projected["rank"] = numeric_rank.astype(int)
    projected = projected.loc[projected["rank"].between(1, k)].copy()
    if projected.empty:
        raise ValueError("ranking has no rows at the requested K")

    group_columns = [*conditions, "qid"]
    expected_ranks = list(range(1, k + 1))
    for key, group in projected.groupby(group_columns, dropna=False, sort=False):
        ranks = sorted(group["rank"].tolist())
        if ranks != expected_ranks:
            raise ValueError(f"{key}: ranking must contain exactly ranks 1 through {k}")
        if group["item_id"].duplicated().any():
            raise ValueError(f"{key}: ranking contains duplicate candidates in top-{k}")

    projected.sort_values([*conditions, "qid", "rank"], inplace=True, kind="stable")
    return projected.reset_index(drop=True)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--condition-column", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an immutable top-K projection")
    receipt_path = args.output.with_suffix(args.output.suffix + ".receipt.json")
    if receipt_path.exists():
        raise FileExistsError("refusing to overwrite an immutable top-K receipt")
    ranking = pd.read_parquet(args.input)
    projected = project_top_k(ranking, k=args.k, condition_columns=args.condition_column)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    projected.to_parquet(args.output, index=False)
    receipt = {
        "schema_version": "frozen-top-k-projection.v1",
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "k": args.k,
        "condition_columns": args.condition_column,
        "rows": len(projected),
        "conditions": int(projected.groupby([*args.condition_column], dropna=False).ngroups),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
