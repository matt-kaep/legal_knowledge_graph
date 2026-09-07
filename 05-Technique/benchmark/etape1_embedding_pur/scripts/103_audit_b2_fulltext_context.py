#!/usr/bin/env python3
"""Audit whether immutable full-text reranking prompts fit a model context."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable


SCRIPTS = Path(__file__).resolve().parent


def _load_reranking_runner():
    spec = importlib.util.spec_from_file_location("b2_reranking_runner_for_context_audit", SCRIPTS / "102_run_b2_comparable_reranking.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_token_counts(
    token_counts: Iterable[tuple[str, int]],
    *,
    context_limit_tokens: int,
    max_output_tokens: int,
) -> dict[str, object]:
    """Report prompt compatibility while reserving the full completion budget."""
    if context_limit_tokens <= 0:
        raise ValueError("context_limit_tokens must be positive")
    if max_output_tokens <= 0 or max_output_tokens >= context_limit_tokens:
        raise ValueError("max_output_tokens must be positive and below the context limit")

    input_token_budget = context_limit_tokens - max_output_tokens
    observed = [(str(qid), int(count)) for qid, count in token_counts]
    if not observed:
        raise ValueError("token_counts must not be empty")
    if any(count < 0 for _, count in observed):
        raise ValueError("token counts must be non-negative")

    max_input_tokens = max(count for _, count in observed)
    overflow_qids = [qid for qid, count in observed if count > input_token_budget]
    return {
        "context_limit_tokens": context_limit_tokens,
        "max_output_tokens": max_output_tokens,
        "input_token_budget": input_token_budget,
        "questions": len(observed),
        "max_input_tokens": max_input_tokens,
        "max_total_tokens": max_input_tokens + max_output_tokens,
        "overflow_qids": overflow_qids,
        "compatible": not overflow_qids,
    }


def count_chat_tokens(tokenizer: Any, prompt: str) -> int:
    """Count the exact chat-template input tokens before generation starts."""
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(token_ids)


def write_immutable_report(path: Path, report: dict[str, object]) -> None:
    """Write one complete audit receipt without replacing prior evidence."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite full-text context audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"job must be a JSON object: {path}")
            jobs.append(value)
    if not jobs:
        raise ValueError(f"jobs file is empty: {path}")
    return jobs


def snapshot_tokenizer_hashes(model_snapshot: Path) -> dict[str, str]:
    required = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    missing = [name for name in required if not (model_snapshot / name).is_file()]
    if missing:
        raise ValueError(f"model snapshot lacks tokenizer evidence: {missing}")
    return {name: sha256(model_snapshot / name) for name in required}


def audit_frozen_job_files(
    job_paths: Iterable[Path],
    *,
    prompt_paths: dict[str, Path],
    tokenizer: Any,
    model_id: str,
    model_revision: str,
    model_snapshot: Path | None = None,
    context_limit_tokens: int,
    max_output_tokens: int,
    expected_questions: int,
) -> dict[str, object]:
    """Audit immutable jobs using the model's generation chat template."""
    if set(prompt_paths) != {"article", "jp"}:
        raise ValueError("prompt_paths must contain exactly article and jp")
    all_jobs: list[dict[str, Any]] = []
    job_files: list[dict[str, object]] = []
    for path in job_paths:
        jobs = _load_jobs(path)
        all_jobs.extend(jobs)
        job_files.append({"path": str(path), "jobs": len(jobs), "sha256": sha256(path)})
    templates = {modality: path.read_text(encoding="utf-8") for modality, path in prompt_paths.items()}
    audit = audit_fulltext_jobs(
        all_jobs,
        prompt_templates=templates,
        count_prompt_tokens=lambda prompt: count_chat_tokens(tokenizer, prompt),
        context_limit_tokens=context_limit_tokens,
        max_output_tokens=max_output_tokens,
        expected_questions=expected_questions,
    )
    model: dict[str, object] = {"id": model_id, "revision": model_revision}
    if model_snapshot is not None:
        model["local_snapshot"] = str(model_snapshot)
        model["tokenizer_files"] = snapshot_tokenizer_hashes(model_snapshot)
    return {
        **audit,
        "model": model,
        "context_limit_tokens": context_limit_tokens,
        "max_output_tokens": max_output_tokens,
        "candidate_text_policy": "full_text_unmodified",
        "job_files": job_files,
        "prompt_files": {
            modality: {"path": str(path), "sha256": sha256(path)}
            for modality, path in sorted(prompt_paths.items())
        },
    }


def load_tokenizer(model_id: str, revision: str, model_snapshot: Path | None = None) -> Any:
    """Load only the tokenizer used by the frozen reranking model revision."""
    from transformers import AutoTokenizer

    if model_snapshot is not None:
        if not model_snapshot.is_dir():
            raise ValueError(f"model snapshot directory does not exist: {model_snapshot}")
        return AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True)
    return AutoTokenizer.from_pretrained(model_id, revision=revision)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, action="append", required=True, help="Immutable JSONL job file; repeat for several depths.")
    parser.add_argument("--prompt-article", type=Path, required=True)
    parser.add_argument("--prompt-jp", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-snapshot", type=Path, help="Local immutable snapshot used by vLLM when the remote revision is unavailable.")
    parser.add_argument("--context-limit-tokens", type=int, default=16384)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--expected-questions", type=int, default=754)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_frozen_job_files(
        args.jobs,
        prompt_paths={"article": args.prompt_article, "jp": args.prompt_jp},
        tokenizer=load_tokenizer(args.model_id, args.model_revision, args.model_snapshot),
        model_id=args.model_id,
        model_revision=args.model_revision,
        model_snapshot=args.model_snapshot,
        context_limit_tokens=args.context_limit_tokens,
        max_output_tokens=args.max_output_tokens,
        expected_questions=args.expected_questions,
    )
    write_immutable_report(args.output, report)
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output), "conditions": len(report["conditions"])}))
    return 0


def audit_fulltext_jobs(
    jobs: Iterable[dict[str, Any]],
    *,
    prompt_templates: dict[str, str],
    count_prompt_tokens: Callable[[str], int],
    context_limit_tokens: int,
    max_output_tokens: int,
    expected_questions: int,
) -> dict[str, object]:
    """Audit frozen jobs by condition without shortening any candidate text."""
    if expected_questions <= 0:
        raise ValueError("expected_questions must be positive")
    runner = _load_reranking_runner()
    grouped: dict[tuple[str, str, int], list[tuple[str, int]]] = {}
    for job in jobs:
        family = str(job["family"])
        modality = str(job["modality"])
        qid = str(job["qid"])
        k_in = int(job["k_in"])
        if modality not in prompt_templates:
            raise ValueError(f"missing prompt template for modality {modality}")
        candidates = job.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != k_in:
            raise ValueError(f"{family}/{modality}/{qid}: candidate count differs from K_in")
        if any(not isinstance(candidate.get("text"), str) or not candidate["text"].strip() for candidate in candidates):
            raise ValueError(f"{family}/{modality}/{qid}: candidate text must be complete and non-empty")
        prompt = runner.render_reranking_prompt(prompt_templates[modality], job)
        grouped.setdefault((family, modality, k_in), []).append((qid, int(count_prompt_tokens(prompt))))

    conditions: list[dict[str, object]] = []
    for (family, modality, k_in), token_counts in sorted(grouped.items()):
        qids = [qid for qid, _ in token_counts]
        if len(token_counts) != expected_questions or len(set(qids)) != expected_questions:
            raise ValueError(f"{family}/{modality}/K={k_in}: expected exactly {expected_questions} unique questions")
        summary = audit_token_counts(
            sorted(token_counts),
            context_limit_tokens=context_limit_tokens,
            max_output_tokens=max_output_tokens,
        )
        conditions.append({
            "family": family,
            "modality": modality,
            "k_in": k_in,
            "candidate_text_policy": "full_text_unmodified",
            **summary,
        })
    if not conditions:
        raise ValueError("jobs must not be empty")
    return {
        "schema_version": "b2-e029-fulltext-context-audit.v1",
        "conditions": conditions,
    }


if __name__ == "__main__":
    raise SystemExit(main())
