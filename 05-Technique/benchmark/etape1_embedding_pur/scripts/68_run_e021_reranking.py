"""E021 comparable reranking runner.

The runner operates on immutable, hashed candidate-pool jobs.  Pool creation is
intentionally separate from model execution so that cosine, PPR and LightGCN
cannot silently share or synthesize candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SCRIPT_DIR))

import metrics as retrieval_metrics  # noqa: E402


class InvalidRerankerResponse(ValueError):
    """The provider response violates the frozen E021 output contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compute_input_sha256(job: dict[str, Any]) -> str:
    """Hash only the model-visible contract, not incidental filesystem metadata."""
    payload = {
        "family": job["family"],
        "qid": job["qid"],
        "modality": job.get("modality", "jp"),
        "question": job["question"],
        "candidate_ids": [candidate["item_id"] for candidate in job["candidates"]],
        "candidates": job["candidates"],
        "k_in": job["k_in"],
        "k_out": job["k_out"],
        "prompt_sha256": job.get("prompt_sha256"),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def parse_ranked_ids(raw: str, pool_ids: Iterable[str], k_out: int) -> list[str]:
    """Parse strict JSON and enforce subset, cardinality and uniqueness."""
    if not isinstance(raw, str) or raw.startswith("``"):
        raise InvalidRerankerResponse("response must be plain JSON, without a code fence")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidRerankerResponse(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"ranked_jp_ids"}:
        raise InvalidRerankerResponse("response keys must be exactly ranked_jp_ids")
    ranked = payload["ranked_jp_ids"]
    if not isinstance(ranked, list) or len(ranked) != k_out:
        raise InvalidRerankerResponse(f"expected exactly {k_out} ranked ids")
    if any(not isinstance(item, str) for item in ranked):
        raise InvalidRerankerResponse("ranked ids must be strings")
    if len(set(ranked)) != len(ranked):
        raise InvalidRerankerResponse("ranked ids must be distinct")
    pool = set(pool_ids)
    if not set(ranked).issubset(pool):
        raise InvalidRerankerResponse("ranked ids must belong to the supplied pool")
    return ranked


def _gold_ids(question: dict[str, Any], modality: str) -> set[str]:
    if modality == "article":
        return {str(value) for value in question.get("articles_attendus", [])}
    return {str(value) for value in question.get("gold_jp_ids", [])}


def compute_metrics(
    questions: Iterable[dict[str, Any]],
    responses: Iterable[dict[str, Any]],
    k: int = 10,
    modality: str = "jp",
) -> dict[str, float]:
    """Macro-average exact retrieval metrics over non-empty gold sets."""
    response_by_qid = {str(record["qid"]): record for record in responses}
    rows: list[dict[str, float]] = []
    for question in questions:
        gold = _gold_ids(question, modality)
        response = response_by_qid.get(str(question["qid"]))
        if not gold or response is None:
            continue
        ranked = [str(value) for value in response["ranked_jp_ids"]]
        rows.append(
            {
                "recall_at_10": retrieval_metrics.m1_recall(ranked, gold, k),
                "official_hit_at_10": retrieval_metrics.hit_at_k(ranked, gold, k),
                "ndcg_at_10": retrieval_metrics.ndcg_at_k(ranked, gold, k),
                "mrr_at_10": retrieval_metrics.mrr_at_k(ranked, gold, k),
                "exact_any_gold_at_10": float(bool(set(ranked[:k]) & gold)),
            }
        )
    if not rows:
        raise ValueError("no question with both gold labels and a valid response")
    frame = pd.DataFrame(rows)
    return {column: float(frame[column].mean()) for column in frame.columns}


def _render_prompt(prompt_template: str, job: dict[str, Any]) -> str:
    candidates = [
        {"item_id": candidate["item_id"], "text": candidate["text"]}
        for candidate in job["candidates"]
    ]
    return (
        f"{prompt_template.rstrip()}\n\n"
        f"Question:\n{job['question']}\n\n"
        f"Candidate pool (preserve identifiers exactly):\n"
        f"{json.dumps(candidates, ensure_ascii=False, indent=2)}"
    )


def call_openai_compatible(
    endpoint: str,
    model: str,
    prompt: str,
    timeout_seconds: int = 300,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
    }
    request = Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"reranker provider request failed: {exc}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("provider response has no choices[0].message.content") from exc
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise RuntimeError("provider content is not text")
    return content


def _latest_valid_responses(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "ok":
                continue
            latest[(str(record["family"]), str(record["qid"]))] = record
    return latest


def run_jobs(
    jobs_path: Path,
    responses_path: Path,
    endpoint: str,
    model: str,
    prompt_template: str,
    retry_invalid: bool = True,
) -> dict[str, int]:
    """Run only absent/stale family-question units and append valid results."""
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _latest_valid_responses(responses_path)
    counts = {"jobs": 0, "skipped": 0, "completed": 0, "failed": 0}
    with jobs_path.open(encoding="utf-8") as jobs, responses_path.open("a", encoding="utf-8") as output:
        for line in jobs:
            if not line.strip():
                continue
            job = json.loads(line)
            counts["jobs"] += 1
            key = (str(job["family"]), str(job["qid"]))
            expected_input_sha = compute_input_sha256(job)
            if (
                key in existing
                and existing[key].get("input_sha256") == expected_input_sha
                and len(existing[key].get("ranked_jp_ids", [])) == job["k_out"]
            ):
                counts["skipped"] += 1
                continue
            attempts = 2 if retry_invalid else 1
            last_error = ""
            for attempt in range(attempts):
                try:
                    raw = call_openai_compatible(
                        endpoint,
                        model,
                        _render_prompt(prompt_template, job),
                    )
                    ranked = parse_ranked_ids(
                        raw,
                        [candidate["item_id"] for candidate in job["candidates"]],
                        job["k_out"],
                    )
                    record = {
                        "experiment_id": job["experiment_id"],
                        "family": job["family"],
                        "qid": job["qid"],
                        "modality": job.get("modality", "jp"),
                        "input_sha256": expected_input_sha,
                        "ranked_jp_ids": ranked,
                        "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                        "attempt": attempt + 1,
                        "status": "ok",
                    }
                    output.write(_canonical_json(record) + "\n")
                    output.flush()
                    existing[key] = record
                    counts["completed"] += 1
                    break
                except Exception as exc:  # keep a resumable record for every attempted unit
                    last_error = str(exc)
                    if attempt + 1 < attempts:
                        continue
                    failure = {
                        "experiment_id": job["experiment_id"],
                        "family": job["family"],
                        "qid": job["qid"],
                        "input_sha256": expected_input_sha,
                        "status": "invalid",
                        "error": last_error,
                    }
                    output.write(_canonical_json(failure) + "\n")
                    output.flush()
                    counts["failed"] += 1
    return counts


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["questions"])


def prepare_jobs(
    question_path: Path,
    pool_paths: dict[str, Path],
    output_path: Path,
    experiment_id: str,
    prompt_sha256: str,
    k_in: int = 20,
    k_out: int = 10,
) -> int:
    questions = {str(question["qid"]): question for question in load_questions(question_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for family, pool_path in pool_paths.items():
            with pool_path.open(encoding="utf-8") as pools:
                for line in pools:
                    if not line.strip():
                        continue
                    pool = json.loads(line)
                    qid = str(pool["qid"])
                    question = questions[qid]
                    candidates = list(pool["candidates"])
                    if len(candidates) != k_in:
                        raise ValueError(f"{family}/{qid}: expected {k_in} candidates")
                    job = {
                        "experiment_id": experiment_id,
                        "family": family,
                        "qid": qid,
                        "modality": pool.get("modality", "jp"),
                        "question": question["enonce"],
                        "candidates": candidates,
                        "k_in": k_in,
                        "k_out": k_out,
                        "prompt_sha256": prompt_sha256,
                    }
                    job["input_sha256"] = compute_input_sha256(job)
                    output.write(_canonical_json(job) + "\n")
                    count += 1
    return count


def _parse_family_paths(values: list[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        family, separator, raw_path = value.partition("=")
        if not separator or not family or not raw_path:
            raise ValueError(f"expected FAMILY=PATH, got {value!r}")
        if family in paths:
            raise ValueError(f"duplicate family {family}")
        paths[family] = Path(raw_path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--pool", action="append", required=True, help="FAMILY=POOL.jsonl")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--experiment-id", default="E021")
    prepare.add_argument("--prompt-sha256", required=True)
    prepare.add_argument("--k-in", type=int, default=20)
    prepare.add_argument("--k-out", type=int, default=10)

    run = subparsers.add_parser("run")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--responses", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--prompt", type=Path, required=True)
    run.add_argument("--no-retry-invalid", action="store_true")

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--questions", type=Path, required=True)
    aggregate.add_argument("--responses", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    aggregate.add_argument("--k", type=int, default=10)
    aggregate.add_argument("--modality", default="jp", choices=["jp", "article"])

    args = parser.parse_args(argv)
    if args.command == "prepare":
        count = prepare_jobs(
            args.questions,
            _parse_family_paths(args.pool),
            args.output,
            args.experiment_id,
            args.prompt_sha256,
            args.k_in,
            args.k_out,
        )
        print(json.dumps({"jobs": count}))
        return 0
    if args.command == "run":
        print(json.dumps(run_jobs(
            args.jobs,
            args.responses,
            args.endpoint,
            args.model,
            args.prompt.read_text(encoding="utf-8"),
            retry_invalid=not args.no_retry_invalid,
        )))
        return 0
    metrics = compute_metrics(
        load_questions(args.questions),
        list(_latest_valid_responses(args.responses).values()),
        k=args.k,
        modality=args.modality,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
