#!/usr/bin/env python3
"""Run and score E027: a question-only direct LLM baseline on the frozen A3 split.

The model sees exactly one immutable prompt and one question.  It never sees a
candidate list, graph, retrieval score, source text, web result or tool.  The
only post-processing is a recorded syntactic resolution against the A3 returnable
candidate universe.  Unknown, ambiguous, duplicate, malformed and missing
references occupy an explicit zero-valued output position.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CODE_REPO = Path(os.environ.get("LKG_REPO", str(ROOT.parents[3]))).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import metrics as retrieval_metrics  # noqa: E402


K_OUT = 10


class InvalidDirectResponse(ValueError):
    """The provider response violates E027's frozen output schema."""


class ReferenceResolver(Protocol):
    def resolve(self, reference: str) -> tuple[str | None, str]: ...


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_unique(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        identifier = str(value)
        if identifier not in seen:
            result.append(identifier)
            seen.add(identifier)
    return result


def stable_sequence_sha256(values: Iterable[object]) -> str:
    return hashlib.sha256(
        json.dumps(stable_unique(values), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _normalise_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().strip(" .").split())


def _normalise_article_number(value: str) -> str:
    compact = unicodedata.normalize("NFKC", value).strip()
    compact = re.sub(r"^([lrd a e])\.\s*", lambda match: match.group(1).replace(" ", "").upper(), compact, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", compact)


class ArticleResolver:
    """Resolve only exact A3 ids or a declared legal-citation grammar.

    Accepted natural form: ``<official code title>, article <article number>``.
    Matching normalizes Unicode, case, terminal periods and whitespace only;
    code names are never guessed and no semantic similarity is computed.
    """

    _citation_pattern = re.compile(r"^(.+?),?\s+article\s+([^\s][^,;]*)$", re.IGNORECASE)

    def __init__(self, candidate_ids: Iterable[str], code_titles: dict[str, str | None]):
        self.candidate_ids = stable_unique(candidate_ids)
        self._exact_ids = set(self.candidate_ids)
        self._by_citation: dict[tuple[str, str], list[str]] = {}
        for item_id in self.candidate_ids:
            slug, separator, article = item_id.partition(":")
            if not separator or not slug or not article:
                raise ValueError(f"malformed A3 Article candidate id: {item_id!r}")
            title = code_titles.get(slug)
            if not title:
                continue
            key = (_normalise_text(title), _normalise_article_number(article))
            self._by_citation.setdefault(key, []).append(item_id)

    def resolve(self, reference: str) -> tuple[str | None, str]:
        if reference in self._exact_ids:
            return reference, "unique"
        match = self._citation_pattern.fullmatch(reference)
        if match is None:
            return None, "unresolved"
        key = (_normalise_text(match.group(1)), _normalise_article_number(match.group(2)))
        matches = self._by_citation.get(key, [])
        if len(matches) == 1:
            return matches[0], "unique"
        if len(matches) > 1:
            return None, "ambiguous"
        return None, "unresolved"


class ExactIdentifierResolver:
    """Accept an opaque jurisprudence identifier only byte-for-byte."""

    def __init__(self, candidate_ids: Iterable[str]):
        self.candidate_ids = set(stable_unique(candidate_ids))

    def resolve(self, reference: str) -> tuple[str | None, str]:
        if reference in self.candidate_ids:
            return reference, "unique"
        return None, "unresolved"


def parse_direct_response(raw: str, *, k_out: int = K_OUT) -> list[str]:
    if not isinstance(raw, str) or raw.lstrip().startswith("```"):
        raise InvalidDirectResponse("response must be plain JSON, without a code fence")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidDirectResponse(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != {"references"}:
        raise InvalidDirectResponse("response keys must be exactly references")
    references = payload["references"]
    if not isinstance(references, list) or len(references) > k_out:
        raise InvalidDirectResponse(f"references must contain between 0 and {k_out} strings")
    if any(not isinstance(reference, str) or not reference for reference in references):
        raise InvalidDirectResponse("references must contain only non-empty strings")
    return references


def resolve_response_slots(raw: str, resolver: ReferenceResolver, *, k_out: int = K_OUT) -> list[dict[str, Any]]:
    references = parse_direct_response(raw, k_out=k_out)
    slots: list[dict[str, Any]] = []
    seen_resolved: set[str] = set()
    for rank in range(1, k_out + 1):
        if rank > len(references):
            slots.append({"rank": rank, "reference": None, "resolved_item_id": None, "resolution": "empty"})
            continue
        reference = references[rank - 1]
        resolved_item_id, resolution = resolver.resolve(reference)
        if resolved_item_id is not None and resolved_item_id in seen_resolved:
            resolved_item_id = None
            resolution = "duplicate"
        if resolved_item_id is not None:
            seen_resolved.add(resolved_item_id)
        slots.append(
            {
                "rank": rank,
                "reference": reference,
                "resolved_item_id": resolved_item_id,
                "resolution": resolution,
            }
        )
    return slots


def render_direct_prompt(prompt_template: str, question: str) -> str:
    """The E027 model-visible content: fixed prompt followed by one question."""
    return f"{prompt_template.rstrip()}\n\nQuestion :\n{question}\n"


def direct_response_format(k_out: int = K_OUT) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "direct_references",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "references": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 0,
                        "maxItems": k_out,
                    }
                },
                "required": ["references"],
                "additionalProperties": False,
            },
        },
    }


def call_openai_compatible(
    *, endpoint: str, model: str, prompt: str, timeout_seconds: int = 300, k_out: int = K_OUT
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": direct_response_format(k_out),
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
        raise RuntimeError(f"direct LLM provider request failed: {exc}") from exc
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("provider response has no choices[0].message.content") from exc
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
    if not isinstance(content, str):
        raise RuntimeError("provider content is not text")
    return content


def input_sha256(job: dict[str, Any]) -> str:
    """Hash exactly the E027 model-visible job and frozen runtime contract."""
    return hashlib.sha256(
        canonical_json(
            {
                "experiment_id": job["experiment_id"],
                "qid": job["qid"],
                "modality": job["modality"],
                "question": job["question"],
                "prompt_sha256": job["prompt_sha256"],
                "model_id": job["model_id"],
                "model_revision": job["model_revision"],
                "temperature": job["temperature"],
                "k_out": job["k_out"],
            }
        ).encode("utf-8")
    ).hexdigest()


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("benchmark must contain a questions list")
    qids = [str(question.get("qid")) for question in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("benchmark contains duplicate qids")
    return questions


def prepare_jobs(
    *, questions_path: Path, output_path: Path, modality: str, prompt_path: Path,
    experiment_id: str, model_id: str, model_revision: str, k_out: int = K_OUT,
) -> int:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable job file: {output_path}")
    prompt_sha = sha256(prompt_path)
    questions = load_questions(questions_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for question in questions:
            text = str(question.get("enonce") or "").strip()
            if not text:
                raise ValueError(f"qid={question.get('qid')}: empty question")
            job = {
                "experiment_id": experiment_id,
                "qid": str(question["qid"]),
                "modality": modality,
                "question": text,
                "prompt_sha256": prompt_sha,
                "model_id": model_id,
                "model_revision": model_revision,
                "temperature": 0,
                "k_out": k_out,
            }
            job["input_sha256"] = input_sha256(job)
            handle.write(canonical_json(job) + "\n")
    return len(questions)


def _latest_terminal_responses(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("status") in {"ok", "invalid"}:
                    latest[str(record["qid"])] = record
    return latest


def run_jobs(
    *, jobs_path: Path, responses_path: Path, endpoint: str, model_id: str,
    prompt_path: Path, resolver: ReferenceResolver, retry_errors: bool = True,
) -> dict[str, int]:
    prompt = prompt_path.read_text(encoding="utf-8")
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _latest_terminal_responses(responses_path)
    counts = {"jobs": 0, "skipped": 0, "completed": 0, "invalid": 0, "error": 0}
    with jobs_path.open(encoding="utf-8") as jobs, responses_path.open("a", encoding="utf-8") as output:
        for line in jobs:
            if not line.strip():
                continue
            job = json.loads(line)
            counts["jobs"] += 1
            expected_sha = input_sha256(job)
            current = existing.get(str(job["qid"]))
            if current is not None and current.get("input_sha256") == expected_sha:
                counts["skipped"] += 1
                continue
            started = time.monotonic()
            try:
                raw = call_openai_compatible(
                    endpoint=endpoint,
                    model=model_id,
                    prompt=render_direct_prompt(prompt, job["question"]),
                    k_out=int(job["k_out"]),
                )
                slots = resolve_response_slots(raw, resolver, k_out=int(job["k_out"]))
                record = {
                    "experiment_id": job["experiment_id"],
                    "qid": job["qid"],
                    "modality": job["modality"],
                    "input_sha256": expected_sha,
                    "raw_response_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "elapsed_seconds": time.monotonic() - started,
                    "slots": slots,
                    "status": "ok",
                }
                counts["completed"] += 1
            except InvalidDirectResponse as exc:
                record = {
                    "experiment_id": job["experiment_id"],
                    "qid": job["qid"],
                    "modality": job["modality"],
                    "input_sha256": expected_sha,
                    "elapsed_seconds": time.monotonic() - started,
                    "error": str(exc),
                    "slots": _zero_slots(int(job["k_out"]), resolution="invalid_response"),
                    "status": "invalid",
                }
                counts["invalid"] += 1
            except Exception as exc:
                record = {
                    "experiment_id": job["experiment_id"],
                    "qid": job["qid"],
                    "modality": job["modality"],
                    "input_sha256": expected_sha,
                    "elapsed_seconds": time.monotonic() - started,
                    "error": str(exc),
                    "status": "error",
                }
                counts["error"] += 1
                if not retry_errors:
                    raise
            output.write(canonical_json(record) + "\n")
            output.flush()
    return counts


def _zero_slots(k_out: int, *, resolution: str) -> list[dict[str, Any]]:
    return [
        {"rank": rank, "reference": None, "resolved_item_id": None, "resolution": resolution}
        for rank in range(1, k_out + 1)
    ]


def _slots_from_response(response: dict[str, Any], k_out: int) -> list[dict[str, Any]]:
    slots = response.get("slots")
    if not isinstance(slots, list) or len(slots) != k_out:
        raise ValueError(f"qid={response.get('qid')}: response does not contain {k_out} slots")
    ranks = [int(slot.get("rank", -1)) for slot in slots]
    if ranks != list(range(1, k_out + 1)):
        raise ValueError(f"qid={response.get('qid')}: response slots do not preserve ranks 1..{k_out}")
    return slots


def materialize_exact_results(
    *, questions: list[dict[str, Any]], responses: list[dict[str, Any]],
    candidate_ids: set[str], modality: str, k_out: int = K_OUT,
) -> tuple[pd.DataFrame, dict[str, float]]:
    by_qid = {str(response["qid"]): response for response in responses if response.get("status") in {"ok", "invalid"}}
    qids = [str(question["qid"]) for question in questions]
    if set(qids) != set(by_qid):
        missing = sorted(set(qids) - set(by_qid))
        extra = sorted(set(by_qid) - set(qids))
        raise ValueError(f"terminal response coverage mismatch: missing={len(missing)}, extra={len(extra)}")
    gold_field = "articles_attendus" if modality == "article" else "gold_jp_ids"
    rows: list[dict[str, Any]] = []
    scored: list[dict[str, float]] = []
    for question in questions:
        qid = str(question["qid"])
        response = by_qid[qid]
        if str(response.get("modality")) != modality:
            raise ValueError(f"qid={qid}: response modality differs from expected {modality}")
        slots = _slots_from_response(response, k_out)
        ranked: list[str | None] = []
        for slot in slots:
            item_id = slot.get("resolved_item_id")
            if item_id is not None:
                item_id = str(item_id)
                if item_id not in candidate_ids:
                    raise ValueError(f"qid={qid}: resolved item outside A3 candidate universe: {item_id}")
            ranked.append(item_id)
            rows.append(
                {
                    "qid": qid,
                    "modality": modality,
                    "rank": int(slot["rank"]),
                    "item_id": item_id,
                    "reference": slot.get("reference"),
                    "resolution": str(slot.get("resolution")),
                    "response_status": str(response["status"]),
                }
            )
        gold = {str(item) for item in question.get(gold_field, [])}
        if not gold:
            raise ValueError(f"qid={qid}: strict {modality} labels are required")
        scored.append(
            {
                "hit_at_10": retrieval_metrics.hit_at_k(ranked, gold, k_out),
                "ndcg_at_10": retrieval_metrics.ndcg_at_k(ranked, gold, k_out),
                "mrr_at_10": retrieval_metrics.mrr_at_k(ranked, gold, k_out),
                "exact_any_gold_at_10": float(bool(set(ranked) & gold)),
            }
        )
    frame = pd.DataFrame(rows)
    scores = pd.DataFrame(scored)
    metrics = {column: float(scores[column].mean()) for column in scores.columns}
    if modality == "article":
        metrics["recall_at_10"] = metrics["hit_at_10"]
    metrics["questions"] = float(len(questions))
    metrics["resolved_slots"] = float(frame["item_id"].notna().sum())
    metrics["zero_slots"] = float(frame["item_id"].isna().sum())
    return frame, metrics


def _data_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else DATA_REPO / path


def a3_candidate_order(a3_payload: dict[str, Any], modality: str) -> list[str]:
    target = "articles" if modality == "article" else "jurisprudence"
    retrieval = a3_payload["candidate_universes"]["retrieval_candidate_universe"][target]
    structural = a3_payload["candidate_universes"]["graph_node_universe"][target]
    representation = np.load(_data_path(retrieval["representation_order_path"]), allow_pickle=True).tolist()
    graph_ids = set(map(str, np.load(_data_path(structural["path"]), allow_pickle=True).tolist()))
    result = stable_unique(value for value in representation if str(value) in graph_ids)
    if len(result) != int(retrieval["unique_ids"]):
        raise ValueError(f"A3 {target} candidate count differs from its manifest")
    if stable_sequence_sha256(result) != retrieval["stable_unique_sequence_sha256"]:
        raise ValueError(f"A3 {target} candidate order differs from its manifest")
    return result


def load_code_titles(config_path: Path | None = None) -> dict[str, str | None]:
    """Read ``ALL_CODES`` without executing config.py (which creates data dirs)."""
    path = config_path or ROOT / "etape1" / "config.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        target = getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id == "ALL_CODES":
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return {str(key): (str(item) if item is not None else None) for key, item in value.items()}
    raise ValueError("could not find static ALL_CODES mapping in config.py")


def write_materialized_results(
    *, questions_path: Path, responses_path: Path, a3_path: Path, modality: str, out_dir: Path
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen output directory: {out_dir}")
    a3_payload = json.loads(a3_path.read_text(encoding="utf-8"))
    candidate_order = a3_candidate_order(a3_payload, modality)
    responses = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rankings, metrics = materialize_exact_results(
        questions=load_questions(questions_path),
        responses=responses,
        candidate_ids=set(candidate_order),
        modality=modality,
    )
    out_dir.mkdir(parents=True)
    ranking_path = out_dir / "rankings_top10.parquet"
    metrics_path = out_dir / "exact_metrics.json"
    receipt_path = out_dir / "materialization_receipt.json"
    rankings.to_parquet(ranking_path, index=False)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "experiment_id": "E027",
        "modality": modality,
        "questions": len(load_questions(questions_path)),
        "candidate_count": len(candidate_order),
        "candidate_order_sha256": stable_sequence_sha256(candidate_order),
        "inputs": {
            "a3_manifest": {"path": str(a3_path), "sha256": sha256(a3_path)},
            "questions": {"path": str(questions_path), "sha256": sha256(questions_path)},
            "responses": {"path": str(responses_path), "sha256": sha256(responses_path)},
        },
        "outputs": {
            "rankings_top10.parquet": sha256(ranking_path),
            "exact_metrics.json": sha256(metrics_path),
        },
        "metrics": metrics,
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def _resolver_for(modality: str, candidate_order: list[str]) -> ReferenceResolver:
    if modality == "article":
        return ArticleResolver(candidate_order, load_code_titles())
    return ExactIdentifierResolver(candidate_order)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--questions", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--modality", choices=("article", "jp"), required=True)
    prepare.add_argument("--prompt", type=Path, required=True)
    prepare.add_argument("--experiment-id", default="E027")
    prepare.add_argument("--model-id", required=True)
    prepare.add_argument("--model-revision", required=True)
    prepare.add_argument("--k-out", type=int, default=K_OUT)
    run = commands.add_parser("run")
    run.add_argument("--jobs", type=Path, required=True)
    run.add_argument("--responses", type=Path, required=True)
    run.add_argument("--a3-manifest", type=Path, required=True)
    run.add_argument("--endpoint", required=True)
    run.add_argument("--model-id", required=True)
    run.add_argument("--prompt", type=Path, required=True)
    run.add_argument("--no-retry-errors", action="store_true")
    materialize = commands.add_parser("materialize")
    materialize.add_argument("--questions", type=Path, required=True)
    materialize.add_argument("--responses", type=Path, required=True)
    materialize.add_argument("--a3-manifest", type=Path, required=True)
    materialize.add_argument("--modality", choices=("article", "jp"), required=True)
    materialize.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        count = prepare_jobs(
            questions_path=args.questions,
            output_path=args.output,
            modality=args.modality,
            prompt_path=args.prompt,
            experiment_id=args.experiment_id,
            model_id=args.model_id,
            model_revision=args.model_revision,
            k_out=args.k_out,
        )
        print(json.dumps({"jobs": count, "output": str(args.output)}))
        return 0
    if args.command == "run":
        a3_payload = json.loads(args.a3_manifest.read_text(encoding="utf-8"))
        first = next((json.loads(line) for line in args.jobs.read_text(encoding="utf-8").splitlines() if line.strip()), None)
        if first is None:
            raise ValueError("job file is empty")
        candidate_order = a3_candidate_order(a3_payload, str(first["modality"]))
        print(json.dumps(run_jobs(
            jobs_path=args.jobs,
            responses_path=args.responses,
            endpoint=args.endpoint,
            model_id=args.model_id,
            prompt_path=args.prompt,
            resolver=_resolver_for(str(first["modality"]), candidate_order),
            retry_errors=not args.no_retry_errors,
        )))
        return 0
    receipt = write_materialized_results(
        questions_path=args.questions,
        responses_path=args.responses,
        a3_path=args.a3_manifest,
        modality=args.modality,
        out_dir=args.out_dir,
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
