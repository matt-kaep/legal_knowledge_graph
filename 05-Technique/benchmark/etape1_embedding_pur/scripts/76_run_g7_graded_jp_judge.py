#!/usr/bin/env python3
"""Run the resumable E016 graded jurisprudence judge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable


REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
ETAPE1 = REPO / "05-Technique/benchmark/etape1_embedding_pur"
PROMPT_PATH = ETAPE1 / "prompts/g7_graded_jp_judge_v1.txt"
SCHEMA_PATH = ETAPE1 / "schemas/g7_graded_jp_judge_v1.json"
DEFAULT_OUT = (
    ETAPE1
    / "data/doctrine_v3plus_bench/G7-citation-JJ-cit1-sem025-knn5"
    / "eval_rich_retrievable_strict/E016-g7-graded-jp-v1"
)
DEFAULT_MODEL = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
DEFAULT_MODEL_REVISION = "519bdca117c8f10a9a578d1b70b5c0d54c59b7ba"


def _load_contract_module():
    path = ETAPE1 / "scripts/74_g7_graded_jp_contract.py"
    spec = importlib.util.spec_from_file_location("g7_graded_jp_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract_module()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_runtime_contract(model_id: str, model_revision: str) -> dict[str, str]:
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_version": "g7_graded_jp_judge_v1",
        "prompt_sha256": file_sha256(PROMPT_PATH),
        "schema_sha256": file_sha256(SCHEMA_PATH),
    }


def load_jobs(path: Path, limit: int | None = None) -> list[dict]:
    paths = [path] if path.is_file() else sorted((path / "jobs").glob("jobs-*.jsonl"))
    if path.is_dir() and not paths:
        direct = path / "judge_jobs.jsonl"
        paths = [direct] if direct.exists() else []
    jobs: list[dict] = []
    for current in paths:
        with current.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                jobs.append(json.loads(line))
                if limit is not None and len(jobs) >= limit:
                    return jobs
    return jobs


def load_done(path: Path, *, retry_non_ok: bool = False) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            job_id = row.get("job_id")
            if job_id and (not retry_non_ok or row.get("status") == "ok"):
                done.add(str(job_id))
    return done


def make_prompt(template: str, job: dict) -> str:
    return template.replace("{question}", str(job["question"])).replace(
        "{decision_card}",
        json.dumps(job["decision_card"], ensure_ascii=False, indent=2),
    )


def extract_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def mock_judge(job: dict) -> dict:
    return {
        "classe": "E",
        "justification": (
            "La fiche ne contient aucune règle ni solution portant sur la question posée; "
            "elle ne permet donc pas d'étayer la réponse."
        ),
    }


def verify_contracts(jobs: list[dict], expected: dict) -> None:
    for job in jobs:
        if job.get("judge_contract") != expected:
            raise ValueError(
                f"judge contract mismatch for {job.get('job_id')}: "
                f"expected={expected}, found={job.get('judge_contract')}"
            )


def run_jobs(
    *,
    jobs_path: Path,
    responses_path: Path,
    judge: Callable[[dict], dict],
    workers: int,
    retry_non_ok: bool,
    limit: int | None,
) -> dict[str, int]:
    loaded_jobs = load_jobs(jobs_path, limit=limit)
    if not loaded_jobs:
        return {"loaded": 0, "skipped": 0, "submitted": 0, "ok": 0, "invalid": 0, "error": 0}
    expected_contract = loaded_jobs[0].get("judge_contract") or {}
    verify_contracts(loaded_jobs, expected_contract)
    done = load_done(responses_path, retry_non_ok=retry_non_ok)
    pending = [job for job in loaded_jobs if str(job["job_id"]) not in done]
    counters = {
        "loaded": len(loaded_jobs),
        "skipped": len(loaded_jobs) - len(pending),
        "submitted": len(pending),
        "ok": 0,
        "invalid": 0,
        "error": 0,
    }
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def run_one(job: dict) -> dict:
        started = time.monotonic()
        base = {
            "job_id": job["job_id"],
            "qid": job["qid"],
            "jp_id": job["jp_id"],
            "judge_contract": expected_contract,
        }
        try:
            payload = judge(job)
            valid, reason = CONTRACT.validate_judgment(payload)
            return {
                **base,
                "status": "ok" if valid else "invalid",
                "invalid_reason": reason,
                "response": payload,
                "latency_seconds": round(time.monotonic() - started, 3),
            }
        except Exception as error:  # noqa: BLE001 - technical errors belong in the append-only log.
            return {
                **base,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "latency_seconds": round(time.monotonic() - started, 3),
            }

    with responses_path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=max(1, workers)
    ) as pool:
        futures = [pool.submit(run_one, job) for job in pending]
        for future in as_completed(futures):
            row = future.result()
            counters[row["status"]] += 1
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
    return counters


def make_vllm_judge(args: argparse.Namespace) -> Callable[[dict], dict]:
    from openai import OpenAI

    template = PROMPT_PATH.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    client = OpenAI(
        base_url=f"http://localhost:{args.port}/v1",
        api_key="EMPTY",
        timeout=args.request_timeout,
        max_retries=0,
    )

    def judge(job: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                response = client.chat.completions.create(
                    model=args.model_id,
                    messages=[
                        {"role": "system", "content": "Tu réponds uniquement en JSON valide."},
                        {"role": "user", "content": make_prompt(template, job)},
                    ],
                    temperature=0,
                    max_tokens=args.max_output_tokens,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "g7_graded_jp_judgment",
                            "schema": schema,
                            "strict": True,
                        },
                    },
                )
                return extract_json_object(response.choices[0].message.content or "")
            except Exception as error:  # noqa: BLE001 - retry local inference failures.
                last_error = error
                if attempt < args.retries:
                    time.sleep(min(10, 2 * (attempt + 1)))
        raise RuntimeError(f"judge_failed: {last_error}")

    return judge


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, default=DEFAULT_OUT / "judge_jobs.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--responses-path", type=Path)
    parser.add_argument("--model-id", default=os.environ.get("MODEL_ID", DEFAULT_MODEL))
    parser.add_argument(
        "--model-revision", default=os.environ.get("REVISION", DEFAULT_MODEL_REVISION)
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "32")))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-non-ok", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    responses_path = args.responses_path or args.out_dir / "judge_responses.jsonl"
    jobs = load_jobs(args.jobs, limit=args.limit)
    if not jobs:
        raise ValueError(f"no jobs found at {args.jobs}")
    expected = build_runtime_contract(args.model_id, args.model_revision)
    verify_contracts(jobs, expected)
    judge = mock_judge if args.mock else make_vllm_judge(args)
    started = time.monotonic()
    counters = run_jobs(
        jobs_path=args.jobs,
        responses_path=responses_path,
        judge=judge,
        workers=args.workers,
        retry_non_ok=args.retry_non_ok,
        limit=args.limit,
    )
    summary = {
        "model_id": args.model_id,
        "responses_path": str(responses_path),
        "judge_contract": expected,
        "counters": counters,
        "mock": args.mock,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    summary_dir = args.out_dir / "run_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"judge-{responses_path.stem}.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
