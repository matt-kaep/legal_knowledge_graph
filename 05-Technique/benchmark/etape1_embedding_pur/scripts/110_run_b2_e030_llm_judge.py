#!/usr/bin/env python3
"""Execute and materialize immutable B2/E030 LLM-as-a-Judge jobs.

This runner deliberately accepts only already-frozen JSONL jobs.  It does not
build rankings, select a retriever, or expose retrieval metadata to the judge.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


LABELS = ("A", "B", "C", "D", "E", "non_jugeable")
GAINS = {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.0, "E": 0.0, "non_jugeable": 0.0}
TERMINAL_STATUSES = {"ok", "invalid", "error"}
REQUIRED_JOB_KEYS = {
    "job_id", "family", "modality", "qid", "position", "question",
    "candidate_id_internal", "document", "model_id", "model_revision", "temperature",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visible_text(job: dict[str, Any]) -> str:
    document = job.get("document")
    if not isinstance(document, dict) or len(document) != 1:
        raise ValueError("E030 job document must contain exactly one visible field")
    field, value = next(iter(document.items()))
    expected = "texte" if job.get("modality") == "article" else "synthese"
    if field != expected or not isinstance(value, str) or not value.strip():
        raise ValueError("E030 job document violates the modality-specific visibility contract")
    return value


def render_judge_prompt(template: str, job: dict[str, Any]) -> str:
    """Render only the question and the candidate content visible to the judge."""
    question = job.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("E030 job requires a non-empty question")
    return template.format(question=question, document=_visible_text(job))


def validate_judgment(payload: Any) -> tuple[bool, str | None]:
    if not isinstance(payload, dict):
        return False, "not_an_object"
    if set(payload) != {"classe", "justification"}:
        return False, "invalid_keys"
    if payload["classe"] not in LABELS:
        return False, "unknown_label"
    if not isinstance(payload["justification"], str) or not payload["justification"].strip():
        return False, "invalid_justification"
    return True, None


def _validate_job(job: dict[str, Any], *, expected_model_id: str | None = None) -> None:
    missing = REQUIRED_JOB_KEYS - set(job)
    if missing:
        raise ValueError(f"E030 job lacks fields {sorted(missing)}")
    if int(job["position"]) not in range(1, 11):
        raise ValueError("E030 job position must be in 1..10")
    if job["temperature"] != 0:
        raise ValueError("E030 temperature must be exactly zero")
    if expected_model_id is not None and job["model_id"] != expected_model_id:
        raise ValueError("E030 job model differs from the frozen execution model")
    _visible_text(job)


def _response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "legal_relevance_judgment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "classe": {"type": "string", "enum": list(LABELS)},
                    "justification": {"type": "string", "minLength": 1},
                },
                "required": ["classe", "justification"],
                "additionalProperties": False,
            },
        },
    }


def call_openai_compatible(*, endpoint: str, model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": _response_format(),
    }
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"judge provider request failed: {exc}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("judge provider response has no choices[0].message.content") from exc
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise RuntimeError("judge provider content is not text")
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge provider returned invalid JSON: {exc}") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_judgments(
    *,
    jobs: list[dict[str, Any]],
    responses_path: Path,
    endpoint: str,
    model_id: str,
    prompt_template: str,
    max_tokens: int,
    max_workers: int,
) -> dict[str, int]:
    """Append terminal responses, preserving any valid previously completed calls."""
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    seen: dict[str, dict[str, Any]] = {}
    if responses_path.exists():
        for row in _load_jsonl(responses_path):
            job_id = str(row.get("job_id"))
            if row.get("status") in TERMINAL_STATUSES:
                if job_id in seen:
                    raise ValueError(f"duplicate terminal E030 response: {job_id}")
                seen[job_id] = row
    unique_jobs: dict[str, dict[str, Any]] = {}
    for job in jobs:
        _validate_job(job, expected_model_id=model_id)
        job_id = str(job["job_id"])
        if job_id in unique_jobs:
            raise ValueError(f"duplicate E030 job id: {job_id}")
        unique_jobs[job_id] = job
    unknown = set(seen) - set(unique_jobs)
    if unknown:
        raise ValueError("response file includes jobs absent from the frozen E030 job list")
    pending = [job for job_id, job in unique_jobs.items() if job_id not in seen]

    def invoke(job: dict[str, Any]) -> dict[str, Any]:
        try:
            judgment = call_openai_compatible(
                endpoint=endpoint,
                model=model_id,
                prompt=render_judge_prompt(prompt_template, job),
                max_tokens=max_tokens,
            )
            valid, reason = validate_judgment(judgment)
            return {
                "job_id": job["job_id"],
                "status": "ok" if valid else "invalid",
                "judgment": judgment if valid else None,
                "validation_reason": reason,
            }
        except Exception as exc:  # The immutable response ledger must record technical failures.
            return {"job_id": job["job_id"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("a", encoding="utf-8") as output, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(invoke, job) for job in pending]
        for future in as_completed(futures):
            output.write(canonical_json(future.result()) + "\n")
            output.flush()
    return {"jobs": len(jobs), "skipped": len(seen), "completed": len(pending)}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_judgments(jobs: list[dict[str, Any]], responses: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    """Materialize exact fixed-K E030 scores from complete terminal responses."""
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable E030 output: {out_dir}")
    expected: dict[str, dict[str, Any]] = {}
    for job in jobs:
        _validate_job(job)
        job_id = str(job["job_id"])
        if job_id in expected:
            raise ValueError(f"duplicate E030 job id: {job_id}")
        expected[job_id] = job
    terminal: dict[str, dict[str, Any]] = {}
    for response in responses:
        if response.get("status") not in TERMINAL_STATUSES:
            continue
        job_id = str(response.get("job_id"))
        if job_id in terminal:
            raise ValueError(f"duplicate terminal E030 response: {job_id}")
        terminal[job_id] = response
    if set(expected) != set(terminal):
        raise ValueError(f"terminal response coverage mismatch: {len(terminal)}/{len(expected)}")

    rows: list[dict[str, Any]] = []
    for job_id, job in sorted(expected.items(), key=lambda item: (item[1]["family"], item[1]["modality"], item[1]["qid"], int(item[1]["position"]))):
        response = terminal[job_id]
        valid, reason = validate_judgment(response.get("judgment")) if response["status"] == "ok" else (False, response["status"])
        label = response["judgment"]["classe"] if valid else "non_jugeable"
        rows.append({
            "family": job["family"], "modality": job["modality"], "qid": job["qid"],
            "position": int(job["position"]), "candidate_id_internal": job["candidate_id_internal"],
            "response_status": response["status"], "label": label, "validation_reason": reason,
            "raw_gain": GAINS[label], "effective_gain": 0.0,
        })

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["family"]), str(row["modality"]), str(row["qid"]))].append(row)
    per_question: list[dict[str, Any]] = []
    for (family, modality, qid), group in sorted(groups.items()):
        group.sort(key=lambda row: int(row["position"]))
        if [row["position"] for row in group] != list(range(1, 11)):
            raise ValueError("each E030 condition must contain exactly positions 1 through 10")
        seen_candidates: set[str] = set()
        for row in group:
            candidate_id = str(row["candidate_id_internal"])
            if candidate_id not in seen_candidates:
                row["effective_gain"] = float(row["raw_gain"])
                seen_candidates.add(candidate_id)
        gain_sum = sum(float(row["effective_gain"]) for row in group)
        per_question.append({"family": family, "modality": modality, "qid": qid, "gain_sum": gain_sum, "judge_score_at_10": gain_sum / 10.0})

    summary_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_question:
        summary_groups[(str(row["family"]), str(row["modality"]))].append(float(row["judge_score_at_10"]))
    summary = [
        {"family": family, "modality": modality, "judge_score_at_10": sum(scores) / len(scores), "questions": len(scores)}
        for (family, modality), scores in sorted(summary_groups.items())
    ]

    out_dir.mkdir(parents=True)
    positions_path = out_dir / "per_position_judgments.jsonl"
    positions_path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    questions_path = out_dir / "per_question_judge_scores.csv"
    _write_csv(questions_path, per_question, ["family", "modality", "qid", "gain_sum", "judge_score_at_10"])
    summary_path = out_dir / "judge_scores_at_10.csv"
    _write_csv(summary_path, summary, ["family", "modality", "judge_score_at_10", "questions"])
    receipt = {
        "schema_version": "b2-e030-judge-materialization.v1",
        "jobs": len(jobs),
        "responses": len(responses),
        "conditions": len(per_question),
        "files": {path.name: sha256(path) for path in (positions_path, questions_path, summary_path)},
    }
    (out_dir / "materialization_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--responses", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--prompt", type=Path, required=True)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--max-workers", type=int, default=1)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--jobs", type=Path, required=True)
    materialize.add_argument("--responses", type=Path, required=True)
    materialize.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    jobs = _load_jsonl(args.jobs)
    if args.command == "run":
        print(json.dumps(run_judgments(
            jobs=jobs, responses_path=args.responses, endpoint=args.endpoint, model_id=args.model_id,
            prompt_template=args.prompt.read_text(encoding="utf-8"), max_tokens=args.max_tokens, max_workers=args.max_workers,
        )))
        return 0
    print(json.dumps(materialize_judgments(jobs, _load_jsonl(args.responses), args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
