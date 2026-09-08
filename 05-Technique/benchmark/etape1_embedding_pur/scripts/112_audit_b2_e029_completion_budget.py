#!/usr/bin/env python3
"""Audit a conservative JSON completion budget for immutable E029 jobs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _condition_key(job: dict[str, Any]) -> tuple[str, str, int, str | None]:
    return (
        str(job["family"]), str(job["modality"]), int(job["k_in"]),
        None if job.get("replay_seed") is None else str(job["replay_seed"]),
    )


def _completion_upper_bound(job: dict[str, Any], count_tokens: Callable[[str], int]) -> int:
    """Bound a `ranked_ids` response by concatenating independently tokenized JSON fragments."""
    candidates = job.get("candidates")
    k_out = int(job["k_out"])
    if not isinstance(candidates, list) or len(candidates) < k_out:
        raise ValueError("E029 job has fewer candidates than its requested output depth")
    identifier_costs = sorted(
        (int(count_tokens(json.dumps(str(candidate["item_id"]), ensure_ascii=False))) for candidate in candidates),
        reverse=True,
    )
    return (
        int(count_tokens('{"ranked_ids":['))
        + sum(identifier_costs[:k_out])
        + max(k_out - 1, 0) * int(count_tokens(","))
        + int(count_tokens("]}"))
    )


def audit_completion_budget(
    jobs: Iterable[dict[str, Any]], *, count_tokens: Callable[[str], int], expected_questions: int,
    max_output_tokens: int,
) -> dict[str, Any]:
    if expected_questions <= 0 or max_output_tokens <= 0:
        raise ValueError("expected_questions and max_output_tokens must be positive")
    grouped: dict[tuple[str, str, int, str | None], list[tuple[str, int]]] = defaultdict(list)
    for job in jobs:
        key = _condition_key(job)
        grouped[key].append((str(job["qid"]), _completion_upper_bound(job, count_tokens)))
    if not grouped:
        raise ValueError("completion audit requires at least one job")
    conditions: list[dict[str, Any]] = []
    for (family, modality, k_in, replay_seed), rows in sorted(grouped.items()):
        qids = [qid for qid, _ in rows]
        if len(rows) != expected_questions or len(set(qids)) != expected_questions:
            raise ValueError(f"{family}/{modality}/K={k_in}: expected {expected_questions} unique questions")
        max_qid, maximum = max(rows, key=lambda item: item[1])
        condition = {
            "family": family,
            "modality": modality,
            "k_in": k_in,
            "questions": len(rows),
            "max_conservative_completion_tokens": maximum,
            "max_qid": max_qid,
            "max_output_tokens": max_output_tokens,
            "compatible": maximum <= max_output_tokens,
        }
        if replay_seed is not None:
            condition["replay_seed"] = replay_seed
        conditions.append(condition)
    incompatible = [row for row in conditions if not row["compatible"]]
    if incompatible:
        raise ValueError(f"completion budget overflow in {len(incompatible)} condition(s)")
    return {
        "schema_version": "b2-e029-completion-budget-audit.v1",
        "max_output_tokens": max_output_tokens,
        "conditions": conditions,
        "compatible_conditions": len(conditions),
        "incompatible_conditions": 0,
    }


def _load_jobs(paths: Iterable[Path]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in paths:
        jobs.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return jobs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, action="append", required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=754)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable completion-budget audit: {args.output}")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_snapshot), local_files_only=True)
    report = audit_completion_budget(
        _load_jobs(args.jobs),
        count_tokens=lambda value: len(tokenizer.encode(value, add_special_tokens=False)),
        expected_questions=args.expected_questions,
        max_output_tokens=args.max_output_tokens,
    )
    report["model_snapshot"] = str(args.model_snapshot)
    report["job_files"] = [{"path": str(path), "sha256": sha256(path)} for path in args.jobs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "conditions": len(report["conditions"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
