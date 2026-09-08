#!/usr/bin/env python3
"""Audit frozen E030 LLM-as-a-Judge jobs before any model call.

The audit counts the exact chat-template input tokens plus the reserved output
budget.  It emits an immutable receipt and never contacts an inference server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _visible_text(job: Dict[str, Any]) -> Optional[str]:
    if job.get("zero_slot") is True:
        if job.get("candidate_id_internal") is not None or job.get("document") is not None:
            raise ValueError("an E030 zero slot must not include a candidate or visible document")
        return None
    modality = job.get("modality")
    expected_field = "texte" if modality == "article" else "synthese" if modality == "jp" else None
    if expected_field is None:
        raise ValueError("E030 job modality must be article or jp")
    document = job.get("document")
    if not isinstance(document, dict) or set(document) != {expected_field}:
        raise ValueError("E030 job document violates the modality-specific visibility contract")
    value = document[expected_field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("E030 visible document must be non-empty text")
    return value


def _render(template: str, job: Dict[str, Any], document: str) -> str:
    question = job.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("E030 job requires a non-empty question")
    return template.format(question=question, document=document)


def audit_jobs(
    *,
    jobs: Iterable[Dict[str, Any]],
    templates: Dict[str, str],
    max_model_tokens: int,
    max_output_tokens: int,
    count_tokens: Callable[[str], int],
) -> Dict[str, Any]:
    """Return a content-free context audit for immutable E030 jobs.

    ``count_tokens`` must include the chat template when this function is used
    for a real model snapshot.  Tests can supply a deterministic counter.
    """
    if max_model_tokens <= 0 or max_output_tokens <= 0:
        raise ValueError("token budgets must be positive")
    if max_output_tokens >= max_model_tokens:
        raise ValueError("completion budget must leave at least one input token")
    if set(templates) != {"article", "jp"} or any(not value for value in templates.values()):
        raise ValueError("exactly one non-empty article and jp template are required")

    job_ids: set[str] = set()
    condition_counts: Counter = Counter()
    condition_incompatible: Counter = Counter()
    total_jobs = zero_slots = model_jobs = incompatible_jobs = 0
    maximum_prompt_tokens = maximum_total_tokens = 0
    examples: List[Dict[str, Any]] = []

    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("E030 jobs must be JSON objects")
        job_id = job.get("job_id")
        family = job.get("family")
        modality = job.get("modality")
        qid = job.get("qid")
        if not all(isinstance(value, str) and value for value in (job_id, family, modality, qid)):
            raise ValueError("E030 job requires non-empty job_id, family, modality and qid")
        if job_id in job_ids:
            raise ValueError("duplicate E030 job id")
        job_ids.add(job_id)
        total_jobs += 1
        condition = (family, modality)
        condition_counts[condition] += 1
        document = _visible_text(job)
        if document is None:
            zero_slots += 1
            continue

        prompt_tokens = count_tokens(_render(templates[modality], job, document))
        if not isinstance(prompt_tokens, int) or prompt_tokens < 0:
            raise ValueError("token counter must return a non-negative integer")
        total_tokens = prompt_tokens + max_output_tokens
        model_jobs += 1
        maximum_prompt_tokens = max(maximum_prompt_tokens, prompt_tokens)
        maximum_total_tokens = max(maximum_total_tokens, total_tokens)
        if total_tokens > max_model_tokens:
            incompatible_jobs += 1
            condition_incompatible[condition] += 1
            if len(examples) < 20:
                examples.append({
                    "job_id": job_id,
                    "family": family,
                    "modality": modality,
                    "qid": qid,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": total_tokens,
                    "over_by_tokens": total_tokens - max_model_tokens,
                })

    if total_jobs == 0:
        raise ValueError("E030 audit requires at least one frozen job")
    conditions = []
    for condition in sorted(condition_counts):
        conditions.append({
            "family": condition[0],
            "modality": condition[1],
            "jobs": condition_counts[condition],
            "incompatible_jobs": condition_incompatible[condition],
            "compatible": condition_incompatible[condition] == 0,
        })
    return {
        "schema_version": "b2-e030-context-audit.v1",
        "model_calls": 0,
        "jobs": total_jobs,
        "model_jobs": model_jobs,
        "zero_slots": zero_slots,
        "max_model_tokens": max_model_tokens,
        "max_output_tokens": max_output_tokens,
        "maximum_prompt_tokens": maximum_prompt_tokens,
        "maximum_total_tokens": maximum_total_tokens,
        "incompatible_jobs": incompatible_jobs,
        "compatible_jobs": model_jobs - incompatible_jobs,
        "compatible": incompatible_jobs == 0,
        "conditions": conditions,
        "incompatible_examples": examples,
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _chat_token_counter(model_snapshot: Path) -> Callable[[str], int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers is required for an E030 context audit") from exc
    tokenizer = AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True)

    def count_tokens(prompt: str) -> int:
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens)

    return count_tokens


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--article-prompt", type=Path, required=True)
    parser.add_argument("--jp-prompt", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--max-model-tokens", type=int, required=True)
    parser.add_argument("--max-output-tokens", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.out_dir.exists():
        raise FileExistsError("refusing to overwrite an immutable E030 context audit")
    for path in (args.jobs, args.article_prompt, args.jp_prompt, args.model_snapshot):
        if not path.exists():
            raise FileNotFoundError(path)
    report = audit_jobs(
        jobs=_load_jsonl(args.jobs),
        templates={
            "article": args.article_prompt.read_text(encoding="utf-8"),
            "jp": args.jp_prompt.read_text(encoding="utf-8"),
        },
        max_model_tokens=args.max_model_tokens,
        max_output_tokens=args.max_output_tokens,
        count_tokens=_chat_token_counter(args.model_snapshot),
    )
    args.out_dir.mkdir(parents=True)
    report_path = args.out_dir / "context_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "b2-e030-context-audit-receipt.v1",
        "model_calls": 0,
        "compatible": report["compatible"],
        "files": {
            "context_audit.json": sha256(report_path),
            "jobs.jsonl": sha256(args.jobs),
            "article_prompt": sha256(args.article_prompt),
            "jp_prompt": sha256(args.jp_prompt),
        },
        "model_snapshot": str(args.model_snapshot),
    }
    receipt_path = args.out_dir / "context_audit_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"compatible": report["compatible"], "receipt": str(receipt_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
