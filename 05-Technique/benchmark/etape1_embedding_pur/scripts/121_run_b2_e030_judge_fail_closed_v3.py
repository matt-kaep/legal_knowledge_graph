#!/usr/bin/env python3
"""Run one frozen E030 shard without turning transport failures into labels."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import json
from pathlib import Path


BASE_PATH = Path(__file__).with_name("110_run_b2_e030_llm_judge.py")
spec = importlib.util.spec_from_file_location("e030_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(*, jobs_path: Path, responses_path: Path, endpoint: str, frozen_model: str, served_model: str, prompt_path: Path, max_tokens: int, max_workers: int) -> dict:
    if responses_path.exists():
        raise FileExistsError("v3 fail-closed runner refuses a pre-existing response ledger")
    jobs = load_jsonl(jobs_path)
    for job in jobs:
        base._validate_job(job, expected_model_id=frozen_model)
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    prompt = prompt_path.read_text(encoding="utf-8")
    zero_slots = [job for job in jobs if job.get("zero_slot") is True]
    model_jobs = [job for job in jobs if job.get("zero_slot") is not True]

    def invoke(job):
        judgment = base.call_openai_compatible(endpoint=endpoint, model=served_model, prompt=base.render_judge_prompt(prompt, job), max_tokens=max_tokens)
        valid, reason = base.validate_judgment(judgment)
        if not valid:
            raise RuntimeError(f"invalid judge response for {job['job_id']}: {reason}")
        return {"job_id": job["job_id"], "status": "ok", "judgment": judgment}

    responses_path.parent.mkdir(parents=True, exist_ok=True)
    with responses_path.open("x", encoding="utf-8") as output:
        for job in zero_slots:
            output.write(base.canonical_json({"job_id": job["job_id"], "status": "zero_slot", "judgment": None, "validation_reason": "unresolved_zero_slot"}) + "\n")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(invoke, job) for job in model_jobs]
            for future in as_completed(futures):
                # Any HTTP/transport exception is re-raised: no error row, no label.
                output.write(base.canonical_json(future.result()) + "\n")
                output.flush()
    return {"jobs": len(jobs), "model_calls": len(model_jobs), "zero_slots": len(zero_slots)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True); parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--endpoint", required=True); parser.add_argument("--frozen-model", required=True); parser.add_argument("--served-model", required=True)
    parser.add_argument("--prompt", type=Path, required=True); parser.add_argument("--max-tokens", type=int, default=256); parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(jobs_path=args.jobs, responses_path=args.responses, endpoint=args.endpoint, frozen_model=args.frozen_model, served_model=args.served_model, prompt_path=args.prompt, max_tokens=args.max_tokens, max_workers=args.max_workers)))
