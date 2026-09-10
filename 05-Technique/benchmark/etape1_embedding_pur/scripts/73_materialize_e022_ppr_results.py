"""Materialize hash-bound PPR @10 exports from a completed E022 audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import stdev
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
import sys

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import metrics as retrieval_metrics  # noqa: E402


TOP_K = 10
TARGETS = (
    ("articles", "articles_strict", "art", "articles_attendus", ("Recall@10", "NDCG@10", "MRR@10")),
    ("jp", "jp", "jp", "gold_jp_ids", ("Hit@10", "NDCG@10", "MRR@10")),
)
METRIC_FUNCTIONS = {
    "Recall@10": ("recall_at_10", retrieval_metrics.m1_recall),
    "Hit@10": ("official_hit_at_10", retrieval_metrics.hit_at_k),
    "NDCG@10": ("ndcg_at_10", retrieval_metrics.ndcg_at_k),
    "MRR@10": ("mrr_at_10", retrieval_metrics.mrr_at_k),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_std(values: list[float]) -> float:
    return float(stdev(values)) if len(values) > 1 else 0.0


def _configuration(champion: dict[str, Any]) -> str:
    return (
        f"{champion['method']}; k_in={champion['k_in']}; "
        f"seed_variant={champion['seed_variant']}; alpha={champion['alpha']}"
    )


def _require_hashes(graph_dir: Path, graph_report: dict[str, Any]) -> None:
    for filename, expected_sha256 in graph_report["files"].items():
        path = graph_dir / filename
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"source_hash_mismatch:{graph_dir.name}:{filename}")


def _question_values(
    rankings: pd.DataFrame,
    champion: dict[str, Any],
    questions_by_qid: dict[str, dict[str, Any]],
    gold_field: str,
) -> tuple[dict[str, list[float]], int]:
    selected = rankings[
        (rankings["method"].astype(str) == str(champion["method"]))
        & (rankings["modality"].astype(str) == str(champion["modality"]))
        & (rankings["k_in"] == champion["k_in"])
    ].copy()
    selected["qid"] = selected["qid"].astype(str)
    if set(selected["qid"]) != set(questions_by_qid):
        raise ValueError(f"ranking_question_set_mismatch:{champion['method']}")
    values = {name: [] for name in METRIC_FUNCTIONS}
    duplicate_questions = 0
    for qid in sorted(questions_by_qid):
        group = selected[selected["qid"] == qid].sort_values("rank")
        ranks = [int(rank) for rank in group["rank"].tolist()]
        if ranks != list(range(1, len(ranks) + 1)) or len(ranks) < TOP_K:
            raise ValueError(f"invalid_ranking_positions:{champion['method']}:{qid}")
        ranked = [str(item) for item in group.head(TOP_K)["item_id"].tolist()]
        if len(set(ranked)) != TOP_K:
            duplicate_questions += 1
        gold = {str(item) for item in questions_by_qid[qid].get(gold_field, [])}
        if not gold:
            raise ValueError(f"missing_gold_labels:{champion['method']}:{qid}")
        for label, (_, metric) in METRIC_FUNCTIONS.items():
            values[label].append(float(metric(ranked, gold, TOP_K)))
    return values, duplicate_questions


def build_result_rows(
    *,
    audit: dict[str, Any],
    final_root: Path,
    questions_by_qid: dict[str, dict[str, Any]],
    audit_sha256: str = "",
) -> list[dict[str, Any]]:
    if audit.get("status") != "complete" or audit.get("top_k_metrics") != TOP_K:
        raise ValueError("E022 audit is not a completed top-10 audit")
    rows: list[dict[str, Any]] = []
    for graph_id, graph_report in sorted(audit["graphs"].items()):
        if graph_report.get("status") != "complete" or graph_report.get("errors"):
            raise ValueError(f"E022 graph not complete:{graph_id}")
        graph_dir = final_root / graph_id
        _require_hashes(graph_dir, graph_report)
        champions = json.loads((graph_dir / "selected_champions.json").read_text(encoding="utf-8"))["ppr"]
        rankings = pd.read_parquet(graph_dir / "rankings.parquet")
        for task, target, modality, gold_field, requested_metrics in TARGETS:
            champion = champions[modality]
            values, duplicate_questions = _question_values(
                rankings, champion, questions_by_qid, gold_field
            )
            if duplicate_questions != graph_report["duplicate_item_questions_in_raw_top_10"][target]:
                raise ValueError(f"duplicate_count_mismatch:{graph_id}:{target}")
            for metric_label in requested_metrics:
                audit_key, _ = METRIC_FUNCTIONS[metric_label]
                mean = float(sum(values[metric_label]) / len(values[metric_label]))
                if not math.isclose(
                    mean, float(graph_report["metrics"][target][audit_key]), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"audit_metric_mismatch:{graph_id}:{target}:{metric_label}")
                rows.append(
                    {
                        "experiment_id": "E022",
                        "graph": graph_id,
                        "method": "PPR",
                        "task": task,
                        "split": "eval_rich_retrievable_strict",
                        "selection_folds": 5,
                        "seeds": "not_applicable",
                        "questions": len(values[metric_label]),
                        "metric": metric_label,
                        "mean": mean,
                        "sample_std": _sample_std(values[metric_label]),
                        "configuration": _configuration(champion),
                        "scientific_status": "audite",
                        "historical_summary_top_k": audit["historical_summary_top_k"],
                        "raw_top10_duplicate_questions": duplicate_questions,
                        "source_audit_sha256": audit_sha256,
                        "source_rankings_sha256": graph_report["files"]["rankings.parquet"],
                        "source_champions_sha256": graph_report["files"]["selected_champions.json"],
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _table_rows(rows: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
    table: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row["task"] != task:
            continue
        key = (str(row["graph"]), str(row["method"]))
        target = table.setdefault(
            key,
            {
                "graph": row["graph"],
                "method": row["method"],
                "questions": row["questions"],
                "configuration": row["configuration"],
                "scientific_status": row["scientific_status"],
                "source_audit_sha256": row["source_audit_sha256"],
                "raw_top10_duplicate_questions": row["raw_top10_duplicate_questions"],
            },
        )
        target[f"{row['metric']}_mean"] = row["mean"]
        target[f"{row['metric']}_sample_std"] = row["sample_std"]
    return [table[key] for key in sorted(table)]


def write_outputs(rows: list[dict[str, Any]], *, output_dir: Path, audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    long_path = output_dir / "internal_eval_ppr_exact.csv"
    articles_path = output_dir / "table_articles_ppr_exact.csv"
    jp_path = output_dir / "table_jp_ppr_exact.csv"
    fieldnames = list(rows[0]) if rows else []
    _write_csv(long_path, rows, fieldnames)
    article_rows = _table_rows(rows, "articles")
    jp_rows = _table_rows(rows, "jp")
    _write_csv(articles_path, article_rows, list(article_rows[0]) if article_rows else [])
    _write_csv(jp_path, jp_rows, list(jp_rows[0]) if jp_rows else [])
    manifest = {
        "schema_version": 1,
        "experiment_id": "E022",
        "materializer_script_sha256": sha256_file(Path(__file__).resolve()),
        "source_audit_status": audit["status"],
        "source_audit_sha256": rows[0]["source_audit_sha256"] if rows else "",
        "rows": len(rows),
        "outputs": {
            path.name: sha256_file(path)
            for path in (long_path, articles_path, jp_path)
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    rows = build_result_rows(
        audit=audit,
        final_root=args.final_root,
        questions_by_qid={str(question["qid"]): question for question in questions},
        audit_sha256=sha256_file(args.audit),
    )
    write_outputs(rows, output_dir=args.output_dir, audit=audit)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
