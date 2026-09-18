#!/usr/bin/env python3
"""Materialize the compact A3/B1 main benchmark tables from a sealed metric CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("results/benchmark-a3-b1/source_g6_metrics_at_10.csv")
DEFAULT_OUT_DIR = Path("results/benchmark-a3-b1")
A3_SHA256 = "c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3"

CONDITIONS = (
    {
        "target": "articles", "source": "cosine", "method": "cosine_BGE_M3", "graph": "G1",
        "frozen_configuration": "direct cosine; top_k_out=100", "seeds": "not_applicable",
    },
    {
        "target": "articles", "source": "ppr", "method": "PPR", "graph": "G6-citation-AA-knn5",
        "frozen_configuration": "k_in=50; seed_variant=both; alpha=0.5", "seeds": "not_applicable",
    },
    {
        "target": "articles", "source": "lightgcn_g6", "method": "LightGCN", "graph": "G6-citation-AA-knn5",
        "frozen_configuration": "K=2; learning_rate=0.0005; lambda_anchor=0.5; replay_epochs=9",
        "seeds": "42;43;44",
    },
    {
        "target": "jurisprudence", "source": "cosine", "method": "cosine_BGE_M3", "graph": "G1",
        "frozen_configuration": "direct cosine; top_k_out=100", "seeds": "not_applicable",
    },
    {
        "target": "jurisprudence", "source": "ppr", "method": "PPR", "graph": "G7-citation-AA-cit1-sem025-knn5",
        "frozen_configuration": "k_in=50; seed_variant=both; alpha=0.5", "seeds": "not_applicable",
    },
    {
        "target": "jurisprudence", "source": "lightgcn_g6", "method": "LightGCN", "graph": "G6-citation-AA-knn5",
        "frozen_configuration": "K=3; learning_rate=0.0005; lambda_anchor=0.5; replay_epochs=5",
        "seeds": "42;43;44",
    },
)
REQUIRED_COLUMNS = {
    "source", "target", "hit_at_10", "hit_at_10_seed_std", "ndcg_at_10",
    "ndcg_at_10_seed_std", "mrr_at_10", "mrr_at_10_seed_std", "seeds",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_row(metrics: pd.DataFrame, condition: dict[str, str]) -> pd.Series:
    rows = metrics.loc[
        metrics["source"].eq(condition["source"])
        & metrics["target"].eq(condition["target"])
    ]
    if len(rows) != 1:
        raise ValueError(
            "Missing or duplicate source row for "
            f"source={condition['source']} target={condition['target']}: {len(rows)}"
        )
    return rows.iloc[0]


def _portable_source_reference(source_metrics: Path, out_dir: Path) -> str:
    """Avoid leaking a workstation path into a public receipt."""
    try:
        return str(source_metrics.relative_to(out_dir))
    except ValueError:
        return "external_unbundled_source"


def materialize_tables(source_metrics: Path, out_dir: Path) -> dict[str, Any]:
    """Write separate Article and jurisprudence tables from the sealed G6 receipt CSV."""
    source_metrics = source_metrics.resolve()
    out_dir = out_dir.resolve()
    # Retain the printed decimals from the sealed source.  These are paper
    # exports, so introducing a second float serialization would needlessly
    # alter the auditable representation of a metric.
    metrics = pd.read_csv(source_metrics, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(metrics.columns)
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {sorted(missing)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    source_sha = sha256(source_metrics)
    for condition in CONDITIONS:
        source_row = _source_row(metrics, condition)
        rows.append(
            {
                "method": condition["method"],
                "graph": condition["graph"],
                "task": condition["target"],
                "hit_at_10": source_row["hit_at_10"],
                "hit_at_10_seed_std": source_row["hit_at_10_seed_std"],
                "ndcg_at_10": source_row["ndcg_at_10"],
                "ndcg_at_10_seed_std": source_row["ndcg_at_10_seed_std"],
                "mrr_at_10": source_row["mrr_at_10"],
                "mrr_at_10_seed_std": source_row["mrr_at_10_seed_std"],
                "seeds": condition["seeds"],
                "frozen_configuration": condition["frozen_configuration"],
                "scientific_status": "confirmatory_internal_evaluation_after_train_cv_freeze",
                "a3_manifest_sha256": A3_SHA256,
                "source_metrics_sha256": source_sha,
            }
        )
    table = pd.DataFrame(rows)
    article_path = out_dir / "main_table_articles.csv"
    jp_path = out_dir / "main_table_jurisprudence.csv"
    if article_path.exists() or jp_path.exists():
        raise FileExistsError("Refusing to overwrite an existing A3/B1 paper table")
    table.loc[table["task"].eq("articles")].to_csv(article_path, index=False)
    table.loc[table["task"].eq("jurisprudence")].to_csv(jp_path, index=False)
    manifest = {
        "schema_version": 1,
        "campaign": "B1_A3_G6_paper_ready_v2",
        "scientific_status": "confirmatory_internal_evaluation_after_train_cv_freeze",
        "a3_manifest_sha256": A3_SHA256,
        "source_metrics_path": _portable_source_reference(source_metrics, out_dir),
        "source_metrics_sha256": source_sha,
        "source_rows": len(metrics),
        "tables": {
            "articles": {"path": article_path.name, "sha256": sha256(article_path), "rows": 3},
            "jurisprudence": {"path": jp_path.name, "sha256": sha256(jp_path), "rows": 3},
        },
        "metric_definition": "Hit@10 = |dedup(R_10(q)) intersect Y(q)| / min(|Y(q)|, 10)",
        "scope": (
            "Main B1/A3 table only. PPR uses its separately frozen train/CV champion per task; "
            "LightGCN is the prespecified G6 paper configuration, averaged across seeds 42/43/44."
        ),
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metrics", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    manifest_path = args.out_dir / "main_table_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing receipt: {manifest_path}")
    manifest = materialize_tables(args.source_metrics, args.out_dir)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "sha256": sha256(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
