#!/usr/bin/env python3
"""Run and materialize B2/E029 comparable reranking on frozen real pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import metrics as retrieval_metrics  # noqa: E402
from b2_reranking_prompt import render_reranking_prompt  # noqa: E402


class InvalidRerankingResponse(ValueError):
    """The provider response violates the frozen B2 E029 output contract."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reranking_response_format(k_out: int, pool_ids: Iterable[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ranked_ids",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "ranked_ids": {
                        "type": "array",
                        "items": {"type": "string", "enum": [str(item) for item in pool_ids]},
                        "minItems": k_out,
                        "maxItems": k_out,
                    }
                },
                "required": ["ranked_ids"],
                "additionalProperties": False,
            },
        },
    }


def parse_ranked_ids(raw: str, pool_ids: Iterable[str], k_out: int) -> list[str]:
    if not isinstance(raw, str) or raw.lstrip().startswith("```"):
        raise InvalidRerankingResponse("response must be plain JSON, without a code fence")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidRerankingResponse(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"ranked_ids"}:
        raise InvalidRerankingResponse("response keys must be exactly ranked_ids")
    ranked = payload["ranked_ids"]
    if not isinstance(ranked, list) or len(ranked) != k_out or any(not isinstance(item, str) for item in ranked):
        raise InvalidRerankingResponse(f"ranked_ids must contain exactly {k_out} strings")
    if not set(ranked).issubset(set(map(str, pool_ids))):
        raise InvalidRerankingResponse("ranked_ids must belong to the supplied pool")
    return ranked


def normalize_ranked_ids(ranked_ids: Iterable[str], pool_ids: Iterable[str], k_out: int) -> tuple[list[str], list[str]]:
    pool = [str(item) for item in pool_ids]
    if len(pool) != len(set(pool)) or len(pool) < k_out:
        raise ValueError("pool must contain at least K_out distinct identifiers")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in ranked_ids:
        item = str(item)
        if item not in seen:
            normalized.append(item)
            seen.add(item)
    fallback: list[str] = []
    for item in pool:
        if len(normalized) == k_out:
            break
        if item not in seen:
            normalized.append(item)
            fallback.append(item)
            seen.add(item)
    if len(normalized) != k_out:
        raise ValueError("could not deterministically complete duplicate response slots")
    return normalized, fallback


def zero_slots(k_out: int, *, resolution: str) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "reference": None, "resolved_item_id": None, "resolution": resolution}
        for rank in range(1, k_out + 1)
    ]


def _legacy_candidate_text_representation(modality: str) -> dict[str, Any]:
    if modality == "article":
        return {"source_field": "texte", "projection": "complete_unmodified"}
    if modality == "jp":
        return {"source_field": "synthese", "projection": "complete_unmodified"}
    raise ValueError(f"unsupported reranking modality: {modality}")


def project_candidates_for_reranking(
    candidates: Iterable[dict[str, Any]],
    *,
    modality: str,
    article_token_cap: int | None,
    tokenizer: Any | None,
    tokenizer_id: str | None,
    tokenizer_revision: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the exact candidate text visible to the B2 reranker.

    The projection is intentionally performed before job hashing and inference.
    JP syntheses remain byte-for-byte unchanged; Article text is decoded from the
    exact frozen tokenizer prefix so the context audit and vLLM receive the same
    string.
    """
    copied = [dict(candidate) for candidate in candidates]
    for candidate in copied:
        if not isinstance(candidate.get("text"), str) or not candidate["text"].strip():
            raise ValueError("reranking candidates require a non-empty text field")

    if article_token_cap is None:
        return copied, _legacy_candidate_text_representation(modality)
    if article_token_cap <= 0:
        raise ValueError("article_token_cap must be positive")
    if modality == "jp":
        return copied, _legacy_candidate_text_representation(modality)
    if modality != "article":
        raise ValueError(f"unsupported reranking modality: {modality}")
    if tokenizer is None or not tokenizer_id or not tokenizer_revision:
        raise ValueError("Article projection requires the exact frozen tokenizer id and revision")

    for candidate in copied:
        token_ids = tokenizer.encode(candidate["text"], add_special_tokens=False)
        projected = tokenizer.decode(
            token_ids[:article_token_cap],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if not isinstance(projected, str) or not projected.strip():
            raise ValueError(f"Article projection is empty for candidate {candidate.get('item_id')}")
        candidate["text"] = projected
    return copied, {
        "source_field": "texte",
        "projection": "token_prefix",
        "tokenizer_id": tokenizer_id,
        "tokenizer_revision": tokenizer_revision,
        "token_cap": article_token_cap,
    }


def job_input_sha256(job: dict[str, Any]) -> str:
    payload = {
        "experiment_id": job["experiment_id"], "family": job["family"], "modality": job["modality"],
        "qid": job["qid"], "question": job["question"], "candidates": job["candidates"],
        "k_in": job["k_in"], "k_out": job["k_out"], "replay_seed": job.get("replay_seed"),
        "prompt_sha256": job["prompt_sha256"],
        "model_id": job["model_id"], "model_revision": job["model_revision"], "temperature": job["temperature"],
    }
    # Historical full-text jobs predate this explicit field; retaining their
    # hash payload intact keeps the archive independently materializable.
    if "candidate_text_representation" in job:
        payload["candidate_text_representation"] = job["candidate_text_representation"]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def call_openai_compatible(*, endpoint: str, model: str, prompt: str, pool_ids: list[str], k_out: int) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": reranking_response_format(k_out, pool_ids),
    }
    request = Request(endpoint.rstrip("/") + "/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"reranking provider request failed: {exc}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("provider response has no choices[0].message.content") from exc
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise RuntimeError("provider content is not text")
    return content


def _load_pool(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"pool is empty: {path}")
    return rows


def prepare_jobs(
    *,
    pools: list[Path],
    prompts: dict[str, Path],
    output_path: Path,
    model_id: str,
    model_revision: str,
    article_token_cap: int | None = None,
    tokenizer: Any | None = None,
    tokenizer_id: str | None = None,
    tokenizer_revision: str | None = None,
) -> int:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable job file: {output_path}")
    prompt_sha = {modality: sha256(path) for modality, path in prompts.items()}
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, str, str]] = set()
    for pool_path in pools:
        for pool in _load_pool(pool_path):
            modality = str(pool["modality"])
            if modality not in prompts:
                raise ValueError(f"missing prompt for {modality}")
            candidates, representation = project_candidates_for_reranking(
                pool["candidates"],
                modality=modality,
                article_token_cap=article_token_cap,
                tokenizer=tokenizer,
                tokenizer_id=tokenizer_id,
                tokenizer_revision=tokenizer_revision,
            )
            job = {
                "experiment_id": "E029", "family": str(pool["family"]), "modality": modality,
                "qid": str(pool["qid"]),
                "question": str(pool["question"]), "candidates": candidates,
                "k_in": int(pool["k_in"]), "k_out": int(pool["k_out"]),
                "replay_seed": str(pool["replay_seed"]) if pool.get("replay_seed") is not None else None,
                "source_method": str(pool["source_method"]), "source_ranking_sha256": str(pool["source_ranking_sha256"]),
                "source_texts_sha256": str(pool["source_texts_sha256"]), "prompt_sha256": prompt_sha[modality],
                "model_id": model_id, "model_revision": model_revision, "temperature": 0,
                "candidate_text_representation": representation,
            }
            key = _key(job)
            if key in seen:
                raise ValueError(f"duplicate frozen pool job: {key}")
            seen.add(key)
            if len(job["candidates"]) != job["k_in"] or len({candidate["item_id"] for candidate in job["candidates"]}) != job["k_in"]:
                raise ValueError(f"{key}: pool cardinality is invalid")
            job["input_sha256"] = job_input_sha256(job)
            jobs.append(job)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for job in sorted(jobs, key=_key):
            handle.write(canonical_json(job) + "\n")
    return len(jobs)


def _key(record: dict[str, Any]) -> tuple[str, str, int, str, str]:
    """Identify one immutable E029 context, not just one retrieval question."""
    replay_seed = record.get("replay_seed")
    return (
        str(record["family"]),
        str(record["modality"]),
        int(record["k_in"]),
        "" if replay_seed is None else str(replay_seed),
        str(record["qid"]),
    )


def _latest_terminal(path: Path) -> dict[tuple[str, str, int, str, str], dict[str, Any]]:
    latest: dict[tuple[str, str, int, str, str], dict[str, Any]] = {}
    if not path.exists():
        return latest
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("status") in {"ok", "invalid"}:
                latest[_key(record)] = record
    return latest


def run_jobs(*, jobs_path: Path, responses_path: Path, endpoint: str, model_id: str, prompts: dict[str, Path]) -> dict[str, int]:
    templates = {modality: path.read_text(encoding="utf-8") for modality, path in prompts.items()}
    terminal = _latest_terminal(responses_path)
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"jobs": 0, "skipped": 0, "completed": 0, "invalid": 0, "error": 0}
    with jobs_path.open(encoding="utf-8") as jobs, responses_path.open("a", encoding="utf-8") as output:
        for line in jobs:
            if not line.strip():
                continue
            job = json.loads(line)
            counts["jobs"] += 1
            key = _key(job)
            expected_sha = job_input_sha256(job)
            prior = terminal.get(key)
            if prior and prior.get("input_sha256") == expected_sha:
                counts["skipped"] += 1
                continue
            started = time.monotonic()
            try:
                pool_ids = [str(candidate["item_id"]) for candidate in job["candidates"]]
                raw = call_openai_compatible(endpoint=endpoint, model=model_id, prompt=render_reranking_prompt(templates[job["modality"]], job), pool_ids=pool_ids, k_out=int(job["k_out"]))
                raw_ids = parse_ranked_ids(raw, pool_ids, int(job["k_out"]))
                ranked_ids, fallback_ids = normalize_ranked_ids(raw_ids, pool_ids, int(job["k_out"]))
                slots = [{"rank": rank, "reference": item, "resolved_item_id": item, "resolution": "unique"} for rank, item in enumerate(ranked_ids, start=1)]
                record = {**{name: job[name] for name in ("experiment_id", "family", "modality", "qid", "k_in", "replay_seed")}, "input_sha256": expected_sha, "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(), "elapsed_seconds": time.monotonic() - started, "slots": slots, "response_normalization": {"duplicate_ids_removed": len(raw_ids) - len(set(raw_ids)), "source_order_fallback_ids": fallback_ids}, "status": "ok"}
                counts["completed"] += 1
            except InvalidRerankingResponse as exc:
                record = {**{name: job[name] for name in ("experiment_id", "family", "modality", "qid", "k_in", "replay_seed")}, "input_sha256": expected_sha, "elapsed_seconds": time.monotonic() - started, "error": str(exc), "slots": zero_slots(int(job["k_out"]), resolution="invalid_response"), "status": "invalid"}
                counts["invalid"] += 1
            except Exception as exc:
                record = {**{name: job[name] for name in ("experiment_id", "family", "modality", "qid", "k_in", "replay_seed")}, "input_sha256": expected_sha, "elapsed_seconds": time.monotonic() - started, "error": str(exc), "status": "error"}
                counts["error"] += 1
            output.write(canonical_json(record) + "\n")
            output.flush()
    return counts


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload
    return {str(question["qid"]): question for question in questions}


def materialize_exact_results(*, questions: dict[str, dict[str, Any]], jobs: list[dict[str, Any]], responses: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable materialization: {out_dir}")
    expected = {_key(job): job for job in jobs}
    terminal = {_key(row): row for row in responses if row.get("status") in {"ok", "invalid"}}
    if set(expected) != set(terminal):
        raise ValueError(f"terminal response coverage mismatch: {len(terminal)}/{len(expected)}")
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for key, job in sorted(expected.items()):
        response = terminal[key]
        if response.get("input_sha256") != job_input_sha256(job):
            raise ValueError(f"stale response input hash: {key}")
        slots = response.get("slots", [])
        if [slot.get("rank") for slot in slots] != list(range(1, int(job["k_out"]) + 1)):
            raise ValueError(f"invalid response rank slots: {key}")
        pool = {str(candidate["item_id"]) for candidate in job["candidates"]}
        ranked = [slot.get("resolved_item_id") for slot in slots]
        if any(item is not None and str(item) not in pool for item in ranked):
            raise ValueError(f"response outside its frozen pool: {key}")
        question = questions.get(key[4])
        if question is None:
            raise ValueError(f"question missing from frozen evaluation: {key[4]}")
        gold_field = "articles_attendus" if key[1] == "article" else "gold_jp_ids"
        gold = {str(item) for item in question.get(gold_field, [])}
        if not gold:
            raise ValueError(f"strict gold labels missing: {key}")
        for slot in slots:
            rows.append({"family": key[0], "modality": key[1], "qid": key[4], "rank": slot["rank"], "item_id": slot.get("resolved_item_id"), "reference": slot.get("reference"), "resolution": slot.get("resolution"), "response_status": response["status"], "k_in": job["k_in"], "replay_seed": job.get("replay_seed")})
        metric = {"family": key[0], "modality": key[1], "qid": key[4], "k_in": job["k_in"], "replay_seed": job.get("replay_seed"), "hit_at_10": retrieval_metrics.hit_at_k(ranked, gold, 10), "ndcg_at_10": retrieval_metrics.ndcg_at_k(ranked, gold, 10), "mrr_at_10": retrieval_metrics.mrr_at_k(ranked, gold, 10), "exact_any_gold_at_10": float(bool(set(ranked) & gold))}
        if key[1] == "article":
            metric["recall_at_10"] = metric["hit_at_10"]
        metric_rows.append(metric)
    rankings = pd.DataFrame(rows)
    per_question = pd.DataFrame(metric_rows)
    aggregate = per_question.groupby(["family", "modality", "k_in", "replay_seed"], as_index=False, dropna=False).mean(numeric_only=True)
    seed_mean = aggregate.groupby(["family", "modality", "k_in"], as_index=False, dropna=False).mean(numeric_only=True)
    seed_mean["seed_count"] = aggregate.groupby(["family", "modality", "k_in"], dropna=False).size().to_numpy()
    out_dir.mkdir(parents=True)
    rankings_path = out_dir / "rankings_top10.parquet"
    per_question_path = out_dir / "per_question_exact_metrics.parquet"
    aggregate_path = out_dir / "exact_metrics.csv"
    seed_mean_path = out_dir / "exact_metrics_seed_mean.csv"
    rankings.to_parquet(rankings_path, index=False)
    per_question.to_parquet(per_question_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    seed_mean.to_csv(seed_mean_path, index=False)
    receipt = {"rankings_top10.parquet": sha256(rankings_path), "per_question_exact_metrics.parquet": sha256(per_question_path), "exact_metrics.csv": sha256(aggregate_path), "exact_metrics_seed_mean.csv": sha256(seed_mean_path), "jobs": len(jobs), "responses": len(terminal)}
    (out_dir / "materialization_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--pool", type=Path, action="append", required=True)
    prepare.add_argument("--prompt-article", type=Path, required=True)
    prepare.add_argument("--prompt-jp", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--article-token-cap", type=int, help="Exact frozen-token prefix for Article candidates; omit only for archived full-text jobs.")
    prepare.add_argument("--tokenizer-snapshot", type=Path, help="Local snapshot of the exact tokenizer revision required by --article-token-cap.")
    run = commands.add_parser("run")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--responses", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--prompt-article", type=Path, required=True)
    run.add_argument("--prompt-jp", type=Path, required=True)
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--questions", type=Path, required=True)
    materialize.add_argument("--jobs", type=Path, required=True)
    materialize.add_argument("--responses", type=Path, required=True)
    materialize.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def load_frozen_tokenizer(model_snapshot: Path) -> Any:
    if not model_snapshot.is_dir():
        raise ValueError(f"tokenizer snapshot directory does not exist: {model_snapshot}")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_snapshot), local_files_only=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        prompts = {"article": args.prompt_article, "jp": args.prompt_jp}
        if (args.article_token_cap is None) != (args.tokenizer_snapshot is None):
            raise ValueError("--article-token-cap and --tokenizer-snapshot must be supplied together")
        tokenizer = load_frozen_tokenizer(args.tokenizer_snapshot) if args.tokenizer_snapshot else None
        print(json.dumps({"jobs": prepare_jobs(
            pools=args.pool,
            prompts=prompts,
            output_path=args.output,
            model_id=args.model_id,
            model_revision=args.model_revision,
            article_token_cap=args.article_token_cap,
            tokenizer=tokenizer,
            tokenizer_id=args.model_id if tokenizer else None,
            tokenizer_revision=args.model_revision if tokenizer else None,
        )}))
        return 0
    if args.command == "run":
        prompts = {"article": args.prompt_article, "jp": args.prompt_jp}
        print(json.dumps(run_jobs(jobs_path=args.jobs, responses_path=args.responses, endpoint=args.endpoint, model_id=args.model_id, prompts=prompts)))
        return 0
    jobs = [json.loads(line) for line in args.jobs.read_text(encoding="utf-8").splitlines() if line.strip()]
    responses = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps(materialize_exact_results(questions=load_questions(args.questions), jobs=jobs, responses=responses, out_dir=args.out_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
