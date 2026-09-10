"""Aggregate E021 metrics without collapsing families sharing the same qid."""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import metrics as retrieval_metrics


def _gold_ids(question: dict[str, Any], modality: str) -> set[str]:
    if modality == "article":
        return {str(value) for value in question.get("articles_attendus", [])}
    return {str(value) for value in question.get("gold_jp_ids", [])}


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_response_history(path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], int]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    invalid_records = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["family"]), str(record["qid"]))
            if record.get("status") != "ok":
                invalid_records += 1
                continue
            latest[key] = record
    return latest, invalid_records


def aggregate_metrics(
    questions: Iterable[dict[str, Any]],
    jobs: Iterable[dict[str, Any]],
    responses: Iterable[dict[str, Any]],
    *,
    k: int = 10,
    modality: str = "jp",
) -> dict[str, Any]:
    question_by_qid = {str(question["qid"]): question for question in questions}
    expected_by_family: dict[str, set[str]] = {}
    for job in jobs:
        expected_by_family.setdefault(str(job["family"]), set()).add(str(job["qid"]))

    response_by_key = {
        (str(response["family"]), str(response["qid"])): response
        for response in responses
        if response.get("status") == "ok"
    }
    output: dict[str, Any] = {"k": k, "modality": modality, "families": {}}
    metric_names = [
        "recall_at_10",
        "official_hit_at_10",
        "ndcg_at_10",
        "mrr_at_10",
        "exact_any_gold_at_10",
    ]
    for family in sorted(expected_by_family):
        qids = expected_by_family[family]
        rows: list[dict[str, float]] = []
        gold_qids = {
            qid
            for qid in qids
            if qid in question_by_qid and _gold_ids(question_by_qid[qid], modality)
        }
        missing_qids = sorted(
            qid for qid in gold_qids if (family, qid) not in response_by_key
        )
        for qid in sorted(gold_qids - set(missing_qids)):
            gold = _gold_ids(question_by_qid[qid], modality)
            ranked = [str(value) for value in response_by_key[(family, qid)]["ranked_jp_ids"]]
            rows.append(
                {
                    "recall_at_10": retrieval_metrics.m1_recall(ranked, gold, k),
                    "official_hit_at_10": retrieval_metrics.hit_at_k(ranked, gold, k),
                    "ndcg_at_10": retrieval_metrics.ndcg_at_k(ranked, gold, k),
                    "mrr_at_10": retrieval_metrics.mrr_at_k(ranked, gold, k),
                    "exact_any_gold_at_10": float(bool(set(ranked[:k]) & gold)),
                }
            )
        metrics = {
            name: float(statistics.fmean(row[name] for row in rows))
            for name in metric_names
        }
        dispersion = {
            name: float(statistics.stdev(row[name] for row in rows))
            if len(rows) > 1
            else 0.0
            for name in metric_names
        }
        output["families"][family] = {
            "expected_questions": len(qids),
            "questions_with_gold": len(gold_qids),
            "valid_responses": len(rows),
            "missing_questions": len(missing_qids),
            "missing_qids": missing_qids,
            "coverage": len(rows) / len(gold_qids) if gold_qids else 0.0,
            "status": "complete" if not missing_qids else "incomplete_missing_responses",
            "metrics": metrics,
            "dispersion": dispersion,
            "dispersion_definition": "sample_standard_deviation_over_valid_questions",
        }
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--modality", choices=["jp", "article"], default="jp")
    args = parser.parse_args(argv)

    questions = json.loads(args.questions.read_text(encoding="utf-8"))["questions"]
    jobs = _load_jobs(args.jobs)
    latest, invalid_records = _load_response_history(args.responses)
    result = aggregate_metrics(
        questions,
        jobs,
        latest.values(),
        k=args.k,
        modality=args.modality,
    )
    result["response_history"] = {
        "latest_valid_unique_keys": len(latest),
        "invalid_records": invalid_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
