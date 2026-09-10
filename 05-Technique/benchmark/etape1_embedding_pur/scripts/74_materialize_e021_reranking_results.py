"""Materialize a hash-bound JP reranking table from a completed E021 receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


FAMILIES = ("cosine_bge_m3", "ppr", "lightgcn")
METRICS = (
    ("Hit@10", "official_hit_at_10"),
    ("NDCG@10", "ndcg_at_10"),
    ("MRR@10", "mrr_at_10"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _configuration(manifest: dict[str, Any]) -> str:
    reranker = manifest["reranker"]
    return (
        f"{reranker['model_id']}; revision={reranker['model_revision']}; "
        f"K_in={manifest['k_in']}; K_out={manifest['k_out']}; "
        f"temperature={reranker['temperature']}; "
        f"parser={reranker['local_parser']}"
    )


def _require_complete(
    metrics: dict[str, Any],
    receipt: dict[str, Any],
    *,
    metrics_sha256: str,
    manifest_sha256: str,
) -> None:
    if receipt.get("status") != "complete":
        raise ValueError("E021 completion receipt is not complete")
    if receipt.get("metrics_sha256") != metrics_sha256:
        raise ValueError("E021 metrics hash does not match completion receipt")
    if receipt.get("resume_manifest_sha256") != manifest_sha256:
        raise ValueError("E021 manifest hash does not match completion receipt")
    if set(metrics.get("families", {})) != set(FAMILIES):
        raise ValueError("E021 metric families do not match the frozen contract")
    if set(receipt.get("families", {})) != set(FAMILIES):
        raise ValueError("E021 receipt families do not match the frozen contract")
    for family in FAMILIES:
        metric_row = metrics["families"][family]
        checks = receipt["families"][family].get("checks", {})
        if (
            metric_row.get("status") != "complete"
            or metric_row.get("missing_questions") != 0
            or metric_row.get("valid_responses") != metric_row.get("expected_questions")
            or metric_row.get("coverage") != 1.0
            or not checks
            or not all(checks.values())
        ):
            raise ValueError(f"E021 family is not fully covered:{family}")


def build_rows(
    *,
    metrics: dict[str, Any],
    receipt: dict[str, Any],
    manifest: dict[str, Any],
    metrics_sha256: str,
    receipt_sha256: str,
    manifest_sha256: str,
) -> list[dict[str, Any]]:
    _require_complete(
        metrics,
        receipt,
        metrics_sha256=metrics_sha256,
        manifest_sha256=manifest_sha256,
    )
    rows: list[dict[str, Any]] = []
    configuration = _configuration(manifest)
    for family in FAMILIES:
        family_metrics = metrics["families"][family]
        for label, key in METRICS:
            rows.append(
                {
                    "experiment_id": "E021",
                    "experiment_run_id": manifest["experiment_id"],
                    "family": family,
                    "method": "LLM reranker",
                    "task": "jp",
                    "split": manifest.get("dataset_split", "eval_rich_retrievable_strict"),
                    "questions": family_metrics["expected_questions"],
                    "metric": label,
                    "mean": family_metrics["metrics"][key],
                    "sample_std": family_metrics["dispersion"][key],
                    "configuration": configuration,
                    "scientific_status": "exploratoire",
                    "source_metrics_sha256": metrics_sha256,
                    "completion_receipt_sha256": receipt_sha256,
                    "resume_manifest_sha256": manifest_sha256,
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _table_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = table.setdefault(
            row["family"],
            {
                "family": row["family"],
                "method": row["method"],
                "questions": row["questions"],
                "configuration": row["configuration"],
                "scientific_status": row["scientific_status"],
                "source_metrics_sha256": row["source_metrics_sha256"],
                "completion_receipt_sha256": row["completion_receipt_sha256"],
            },
        )
        target[f"{row['metric']}_mean"] = row["mean"]
        target[f"{row['metric']}_sample_std"] = row["sample_std"]
    return [table[family] for family in FAMILIES]


def write_outputs(
    rows: list[dict[str, Any]], *, output_dir: Path, receipt: dict[str, Any], manifest: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "internal_eval_jp_reranking_exact.csv"
    table_path = output_dir / "table_jp_reranking_exact.csv"
    _write_csv(long_path, rows, list(rows[0]))
    table_rows = _table_rows(rows)
    _write_csv(table_path, table_rows, list(table_rows[0]))
    output_manifest = {
        "schema_version": 1,
        "experiment_id": "E021",
        "experiment_run_id": manifest["experiment_id"],
        "completion_status": receipt["status"],
        "materializer_script_sha256": sha256_file(Path(__file__).resolve()),
        "rows": len(rows),
        "outputs": {
            long_path.name: sha256_file(long_path),
            table_path.name: sha256_file(table_path),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    manifest = json.loads(args.resume_manifest.read_text(encoding="utf-8"))
    rows = build_rows(
        metrics=metrics,
        receipt=receipt,
        manifest=manifest,
        metrics_sha256=sha256_file(args.metrics),
        receipt_sha256=sha256_file(args.receipt),
        manifest_sha256=sha256_file(args.resume_manifest),
    )
    write_outputs(rows, output_dir=args.output_dir, receipt=receipt, manifest=manifest)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
