#!/usr/bin/env python3
"""Orchestrate the E016 G7 graded jurisprudence campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO = Path(
    os.environ.get(
        "LKG_REPO",
        "/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph",
    )
)
ETAPE1 = REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ETAPE1 / "scripts"
DATA = ETAPE1 / "data/doctrine_v3plus_bench"
G7 = "G7-citation-JJ-cit1-sem025-knn5"
EVAL_SPLIT = "eval_rich_retrievable_strict"
DEFAULT_OUT = DATA / G7 / EVAL_SPLIT / "E016-g7-graded-jp-v1"
DEFAULT_RANKINGS = DATA / G7 / EVAL_SPLIT / "rankings.parquet"
DEFAULT_BENCH = DATA / EVAL_SPLIT / "bench_global.json"
DEFAULT_METHOD = "LightGCN-trained_K2"
EXPECTED_QUESTIONS = 754
EXPECTED_POSITIONS = 7_540


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolved_ok_jobs(rows: list[dict]) -> set[str]:
    attempts: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("job_id"):
            attempts[str(row["job_id"])].append(row)
    return {
        job_id
        for job_id, job_attempts in attempts.items()
        if any(row.get("status") == "ok" for row in job_attempts)
    }


def campaign_status(
    out_dir: Path,
    *,
    expected_questions: int = EXPECTED_QUESTIONS,
    expected_positions: int = EXPECTED_POSITIONS,
) -> dict:
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.exists():
        return {
            "experiment_id": "E016",
            "scientific_status": "exploratory_internal_evaluation",
            "preparation_gate": "pending",
            "judge_gate": "pending",
            "summary_gate": "pending",
            "lawyer_gate": "pending",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    preparation_ok = (
        manifest.get("experiment_id") == "E016"
        and int(manifest.get("n_questions", -1)) == expected_questions
        and int(manifest.get("n_positions", -1)) == expected_positions
    )
    response_path = out_dir / "judge_responses.jsonl"
    responses = _read_jsonl(response_path)
    ok_jobs = _resolved_ok_jobs(responses)
    expected_jobs = int(manifest.get("n_jobs", 0))
    if expected_jobs and len(ok_jobs) == expected_jobs:
        judge_gate = "passed"
    elif responses:
        judge_gate = "blocked_technical_incompleteness"
    else:
        judge_gate = "pending"

    summary_path = out_dir / "graded_jp_summary.json"
    summary_gate = "passed" if summary_path.exists() and judge_gate == "passed" else "pending"
    agreement_path = out_dir / "lawyer_audit/lawyer_agreement.json"
    lawyer_gate = "pending"
    lawyer_status = None
    if agreement_path.exists():
        lawyer = json.loads(agreement_path.read_text(encoding="utf-8"))
        lawyer_status = lawyer.get("status")
        if lawyer_status == "complete" and int(lawyer.get("n_annotations", 0)) == 100:
            lawyer_gate = str(lawyer.get("gate_status") or "failed")

    return {
        "experiment_id": "E016",
        "scientific_status": "exploratory_internal_evaluation",
        "preparation_gate": "passed" if preparation_ok else "failed",
        "judge_gate": judge_gate,
        "summary_gate": summary_gate,
        "lawyer_gate": lawyer_gate,
        "prepared_questions": int(manifest.get("n_questions", 0)),
        "prepared_positions": int(manifest.get("n_positions", 0)),
        "duplicate_positions": int(manifest.get("n_duplicate_positions", 0)),
        "expected_judge_jobs": expected_jobs,
        "judge_ok_jobs": len(ok_jobs),
        "lawyer_status": lawyer_status,
    }


def _env_contract_status() -> dict:
    path = REPO / "05-Technique/.env.local"
    required = {"OVH_DB_HOST", "OVH_DB_PORT", "OVH_DB_NAME", "OVH_DB_USER", "OVH_DB_PASSWORD"}
    present: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                if value.strip().strip('"').strip("'"):
                    present.add(key.strip())
    return {"path_exists": path.exists(), "required_keys_present": required <= present}


def preflight() -> dict:
    checks: dict[str, object] = {
        "rankings_exists": DEFAULT_RANKINGS.exists(),
        "benchmark_exists": DEFAULT_BENCH.exists(),
        "prompt_exists": (ETAPE1 / "prompts/g7_graded_jp_judge_v1.txt").exists(),
        "schema_exists": (ETAPE1 / "schemas/g7_graded_jp_judge_v1.json").exists(),
        "db_config": _env_contract_status(),
    }
    if DEFAULT_BENCH.exists():
        bench = json.loads(DEFAULT_BENCH.read_text(encoding="utf-8"))
        checks["benchmark_questions"] = len(bench.get("questions", []))
    if DEFAULT_RANKINGS.exists():
        rankings = pd.read_parquet(DEFAULT_RANKINGS)
        selected = rankings.loc[
            (rankings["method"].astype(str) == DEFAULT_METHOD)
            & (rankings["modality"].astype(str) == "jp")
            & (rankings["rank"].astype(int) <= 10)
        ]
        checks["ranking_questions"] = int(selected["qid"].nunique())
        checks["ranking_positions"] = int(len(selected))
        checks["duplicate_positions"] = int(
            selected.duplicated(["qid", "item_id"], keep="first").sum()
        )
        checks["rank_sequence_ok"] = bool(
            selected.groupby("qid")["rank"].apply(lambda values: sorted(values.astype(int)) == list(range(1, 11))).all()
        )
    ok = (
        checks.get("benchmark_questions") == EXPECTED_QUESTIONS
        and checks.get("ranking_questions") == EXPECTED_QUESTIONS
        and checks.get("ranking_positions") == EXPECTED_POSITIONS
        and checks.get("rank_sequence_ok") is True
        and checks["prompt_exists"] is True
        and checks["schema_exists"] is True
        and checks["db_config"]["required_keys_present"] is True
    )
    return {
        "experiment_id": "E016",
        "scientific_status": "exploratory_internal_evaluation",
        "ok": ok,
        "checks": checks,
    }


def _run_script(script: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *arguments], check=True, cwd=REPO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("status")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--force", action="store_true")
    subparsers.add_parser("summarize")
    audit_parser = subparsers.add_parser("select-lawyer-audit")
    audit_parser.add_argument("--seed", type=int, default=42)
    audit_parser.add_argument("--sample-size", type=int, default=100)
    audit_parser.add_argument("--corpus", type=Path, action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        result = preflight()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 2
    if args.command == "status":
        print(json.dumps(campaign_status(args.out_dir), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare":
        command_args = ["--profile", "evaluation", "--out-dir", str(args.out_dir)]
        if args.force:
            command_args.append("--force")
        _run_script("75_prepare_g7_graded_jp_eval.py", command_args)
        return 0
    if args.command == "summarize":
        _run_script("77_summarize_g7_graded_jp_eval.py", ["--out-dir", str(args.out_dir)])
        return 0
    if args.command == "select-lawyer-audit":
        command_args = [
            "--out-dir",
            str(args.out_dir),
            "--seed",
            str(args.seed),
            "--sample-size",
            str(args.sample_size),
        ]
        for corpus in args.corpus:
            command_args.extend(["--corpus", str(corpus)])
        _run_script("78_select_g7_graded_jp_lawyer_audit.py", command_args)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
