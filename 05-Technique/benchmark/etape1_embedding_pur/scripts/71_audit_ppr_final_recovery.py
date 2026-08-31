"""Audit immutable PPR final-replay outputs before recovery or publication.

This script never writes into a PPR result directory.  It independently checks
the selected train-only champions, the internal-evaluation summary, and the
rankings used to compute the reported top-10 metrics.  A historical result
manifest hash is accepted only when it is declared in a separate recovery
manifest; this prevents a directory with unknown provenance from looking
verified merely because its CSV files are well formed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import metrics as retrieval_metrics  # noqa: E402


REQUIRED_FILES = (
    "selected_champions.json",
    "final_champions_summary.csv",
    "rankings.parquet",
)
TARGETS = {"articles_strict": "art", "jp": "jp"}
TOP_K = 10


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one file without normalizing its bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _float_equal(left: Any, right: float, *, tolerance: float = 1e-12) -> bool:
    try:
        value = float(left)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(value, right, rel_tol=0.0, abs_tol=tolerance)


def _gold_ids(question: dict[str, Any], modality: str) -> set[str]:
    field = "articles_attendus" if modality == "art" else "gold_jp_ids"
    return {str(value) for value in question.get(field, [])}


def _champion_errors(
    champion: dict[str, Any],
    *,
    modality: str,
    expected_fold_sha256: str,
    expected_question_count: int,
) -> list[str]:
    errors: list[str] = []
    if champion.get("modality") != modality:
        errors.append(f"champion_modality_mismatch:{modality}")
    if champion.get("protocol_version") != "grouped_v2":
        errors.append("champion_protocol_version_mismatch")
    if not champion.get("eligible_champion"):
        errors.append("champion_not_eligible")
    if champion.get("n_folds_covered") != 5 or champion.get("expected_folds") != 5:
        errors.append("champion_fold_count_mismatch")
    if champion.get("fold_coverage") != 1.0:
        errors.append("champion_incomplete_fold_coverage")
    if champion.get("n_questions_expected") != expected_question_count:
        errors.append("champion_expected_question_count_mismatch")
    if champion.get("n_questions_covered") != expected_question_count:
        errors.append("champion_question_coverage_mismatch")
    if champion.get("question_coverage") != 1.0:
        errors.append("champion_incomplete_question_coverage")
    if champion.get("fold_assignment_sha256") != expected_fold_sha256:
        errors.append("champion_fold_sha256_mismatch")
    return errors


def _ranking_for_champion(
    rankings: pd.DataFrame,
    champion: dict[str, Any],
) -> pd.DataFrame:
    required = {"qid", "method", "k_in", "modality", "rank", "item_id"}
    missing = sorted(required - set(rankings.columns))
    if missing:
        raise ValueError("rankings_missing_columns:" + ",".join(missing))
    return rankings[
        (rankings["method"].astype(str) == str(champion["method"]))
        & (rankings["modality"].astype(str) == str(champion["modality"]))
        & (rankings["k_in"] == champion["k_in"])
    ].copy()


def _validate_rankings_and_metrics(
    rankings: pd.DataFrame,
    champion: dict[str, Any],
    questions_by_qid: dict[str, dict[str, Any]],
) -> tuple[list[str], int, dict[str, float]]:
    errors: list[str] = []
    selected = _ranking_for_champion(rankings, champion)
    selected["qid"] = selected["qid"].astype(str)
    expected_qids = set(questions_by_qid)
    actual_qids = set(selected["qid"])
    if actual_qids != expected_qids:
        errors.append("ranking_question_set_mismatch")
    metric_rows: list[dict[str, float]] = []
    depths: list[int] = []
    for qid in sorted(expected_qids):
        group = selected[selected["qid"] == qid].sort_values("rank")
        ranks = [int(value) for value in group["rank"].tolist()]
        if ranks != list(range(1, len(ranks) + 1)):
            errors.append(f"ranking_non_contiguous_ranks:{qid}")
            continue
        depths.append(len(ranks))
        if len(ranks) < TOP_K:
            errors.append(f"ranking_shorter_than_top_{TOP_K}:{qid}")
            continue
        top_items = [str(value) for value in group.head(TOP_K)["item_id"].tolist()]
        if len(set(top_items)) != TOP_K:
            errors.append(f"ranking_duplicate_item_in_top_{TOP_K}:{qid}")
            continue
        gold = _gold_ids(questions_by_qid[qid], str(champion["modality"]))
        if not gold:
            errors.append(f"missing_gold_labels:{qid}")
            continue
        metric_rows.append(
            {
                "recall_at_10": retrieval_metrics.m1_recall(top_items, gold, TOP_K),
                "official_hit_at_10": retrieval_metrics.hit_at_k(top_items, gold, TOP_K),
                "mrr_at_10": retrieval_metrics.mrr_at_k(top_items, gold, TOP_K),
                "ndcg_at_10": retrieval_metrics.ndcg_at_k(top_items, gold, TOP_K),
            }
        )
    if not metric_rows:
        errors.append("no_metric_rows")
        metrics = {}
    else:
        metrics = {
            name: float(sum(row[name] for row in metric_rows) / len(metric_rows))
            for name in metric_rows[0]
        }
    return errors, min(depths) if depths else 0, metrics


def audit_ppr_final_outputs(
    *,
    final_root: Path,
    questions_by_qid: dict[str, dict[str, Any]],
    graph_matrix_sha256s: dict[str, str],
    expected_eval_sha256: str,
    expected_fold_sha256: str,
    allowed_result_manifest_sha256s: set[str],
    expected_question_count: int,
) -> dict[str, Any]:
    """Return a fully explicit audit for every graph in a PPR recovery root."""
    report: dict[str, Any] = {
        "audit_kind": "ppr_final_recovery",
        "expected_question_count": expected_question_count,
        "top_k_metrics": TOP_K,
        "graphs": {},
    }
    for graph_id in sorted(graph_matrix_sha256s):
        graph_dir = final_root / graph_id
        graph_report: dict[str, Any] = {
            "status": "complete",
            "errors": [],
            "files": {},
            "metrics": {},
        }
        report["graphs"][graph_id] = graph_report
        missing = [name for name in REQUIRED_FILES if not (graph_dir / name).is_file()]
        if missing:
            graph_report["errors"].append("missing_required_files:" + ",".join(missing))
            graph_report["status"] = "incomplete_or_invalid"
            continue
        graph_report["files"] = {
            name: sha256_file(graph_dir / name) for name in REQUIRED_FILES
        }
        try:
            champions = json.loads((graph_dir / "selected_champions.json").read_text(encoding="utf-8"))
            ppr = champions.get("ppr")
            if not isinstance(ppr, dict):
                raise ValueError("selected_champions_missing_ppr")
            summary_rows = list(csv.DictReader((graph_dir / "final_champions_summary.csv").open(encoding="utf-8")))
            rankings = pd.read_parquet(graph_dir / "rankings.parquet")
            depths: list[int] = []
            for target, modality in TARGETS.items():
                champion = ppr.get(modality)
                if not isinstance(champion, dict):
                    graph_report["errors"].append(f"missing_ppr_champion:{modality}")
                    continue
                graph_report["errors"].extend(
                    _champion_errors(
                        champion,
                        modality=modality,
                        expected_fold_sha256=expected_fold_sha256,
                        expected_question_count=expected_question_count,
                    )
                )
                matching_rows = [
                    row for row in summary_rows
                    if row.get("family") == "ppr"
                    and row.get("target") == target
                    and row.get("method") == str(champion["method"])
                ]
                if len(matching_rows) != 1:
                    graph_report["errors"].append(f"summary_row_count_mismatch:{target}")
                    continue
                row = matching_rows[0]
                if row.get("protocol_version") != "grouped_v2":
                    graph_report["errors"].append(f"summary_protocol_version_mismatch:{target}")
                if row.get("internal_eval_sha256") != expected_eval_sha256:
                    graph_report["errors"].append(f"summary_eval_sha256_mismatch:{target}")
                if row.get("fold_assignment_sha256") != expected_fold_sha256:
                    graph_report["errors"].append(f"summary_fold_sha256_mismatch:{target}")
                if row.get("graph_matrix_sha256") != graph_matrix_sha256s[graph_id]:
                    graph_report["errors"].append(f"summary_matrix_sha256_mismatch:{target}")
                if row.get("manifest_sha256") not in allowed_result_manifest_sha256s:
                    graph_report["errors"].append("undeclared_result_manifest_sha256")
                for key, value in {
                    "question_coverage": 1.0,
                    "n_questions_covered": float(expected_question_count),
                    "n_questions_benchmark": float(expected_question_count),
                    "n_folds_covered": 5.0,
                }.items():
                    if not _float_equal(row.get(key), value):
                        graph_report["errors"].append(f"summary_{key}_mismatch:{target}")
                ranking_errors, depth, metrics = _validate_rankings_and_metrics(
                    rankings, champion, questions_by_qid
                )
                graph_report["errors"].extend(ranking_errors)
                depths.append(depth)
                metric_column_map = {
                    "recall_at_10": "m1",
                    "official_hit_at_10": "hit",
                    "mrr_at_10": "mrr",
                    "ndcg_at_10": "ndcg",
                }
                for metric_name, summary_column in metric_column_map.items():
                    if metrics and not _float_equal(row.get(summary_column), metrics[metric_name]):
                        graph_report["errors"].append(
                            f"summary_metric_mismatch:{target}:{metric_name}"
                        )
                graph_report["metrics"][target] = metrics
            graph_report["ranking_depth"] = min(depths) if depths else 0
        except Exception as exc:
            graph_report["errors"].append(f"audit_exception:{type(exc).__name__}:{exc}")
        if graph_report["errors"]:
            graph_report["status"] = "incomplete_or_invalid"
    report["status"] = (
        "complete"
        if all(graph["status"] == "complete" for graph in report["graphs"].values())
        else "incomplete_or_invalid"
    )
    return report


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    campaign = json.loads(args.campaign_manifest.read_text(encoding="utf-8"))
    recovery = json.loads(args.recovery_manifest.read_text(encoding="utf-8"))
    final_root = args.data_root / recovery["ppr_final_root"]
    questions_path = args.data_root / campaign["datasets"]["internal_eval"]["path"]
    questions = json.loads(questions_path.read_text(encoding="utf-8"))["questions"]
    report = audit_ppr_final_outputs(
        final_root=final_root,
        questions_by_qid={str(question["qid"]): question for question in questions},
        graph_matrix_sha256s={
            str(graph["graph_id"]): str(graph["matrix_sha256"])
            for graph in campaign["graphs"]
        },
        expected_eval_sha256=str(campaign["datasets"]["internal_eval"]["sha256"]),
        expected_fold_sha256=str(campaign["folds"]["sha256"]),
        allowed_result_manifest_sha256s=set(recovery["accepted_result_manifest_sha256s"]),
        expected_question_count=int(campaign["datasets"]["internal_eval"]["questions"]),
    )
    report.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaign_manifest_sha256": _canonical_json_sha256(campaign),
            "campaign_manifest_file_sha256": sha256_file(args.campaign_manifest),
            "recovery_manifest_sha256": sha256_file(args.recovery_manifest),
            "audit_script_sha256": sha256_file(Path(__file__)),
            "ppr_final_root": str(final_root),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
