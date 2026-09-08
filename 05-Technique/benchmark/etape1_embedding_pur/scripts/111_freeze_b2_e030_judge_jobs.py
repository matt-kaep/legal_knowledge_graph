#!/usr/bin/env python3
"""Freeze metadata-blind E030 LLM-as-a-Judge jobs from one final ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("questions file must contain a questions list")
    questions: dict[str, str] = {}
    for row in rows:
        qid = str(row.get("qid") or "")
        question = str(row.get("enonce") or "").strip()
        if not qid or not question or qid in questions:
            raise ValueError("questions require unique qid values and non-empty enonce values")
        questions[qid] = question
    return questions


def _load_cards(path: Path, modality: str) -> dict[str, dict[str, str]]:
    id_column, text_column = ("pair_key", "texte") if modality == "article" else ("jp_id", "synthese")
    frame = pd.read_parquet(path)
    missing = {id_column, text_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{modality} card source lacks columns: {sorted(missing)}")
    frame = frame[[id_column, text_column]].copy()
    frame[id_column] = frame[id_column].astype(str)
    frame[text_column] = frame[text_column].fillna("").astype(str)
    if (frame[text_column].str.strip() == "").any():
        raise ValueError(f"{modality} card source contains an empty visible field")
    if (frame.groupby(id_column)[text_column].nunique() > 1).any():
        raise ValueError(f"{modality} card source contains conflicting content for one identifier")
    field = "texte" if modality == "article" else "synthese"
    return {identifier: {field: text} for identifier, text in frame.drop_duplicates(id_column).itertuples(index=False, name=None)}


def _select_and_validate_ranking(ranking_path: Path, *, questions: dict[str, str], allowed_ids: set[str], modality: str) -> pd.DataFrame:
    ranking = pd.read_parquet(ranking_path)
    missing = {"qid", "rank", "item_id"} - set(ranking.columns)
    if missing:
        raise ValueError(f"ranking lacks columns: {sorted(missing)}")
    if "modality" in ranking.columns:
        aliases = {"article": {"article", "art"}, "jp": {"jp", "jurisprudence"}}
        ranking = ranking.loc[ranking["modality"].astype(str).isin(aliases[modality])].copy()
    ranking["qid"] = ranking["qid"].astype(str)
    ranking["item_id"] = ranking["item_id"].astype(str)
    ranking["rank"] = pd.to_numeric(ranking["rank"], errors="raise").astype(int)
    if set(ranking["qid"]) != set(questions):
        raise ValueError("ranking question coverage differs from the frozen question set")
    expected_ranks = list(range(1, 11))
    for qid, group in ranking.groupby("qid", sort=False):
        ranks = sorted(group["rank"].tolist())
        if ranks != expected_ranks:
            raise ValueError(f"{qid}: ranking must contain exactly ranks 1 through 10")
        candidate_ids = group["item_id"].tolist()
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"{qid}: ranking contains duplicate candidates")
    outside = set(ranking["item_id"]) - allowed_ids
    if outside:
        raise ValueError("ranking contains candidates outside the frozen A3 candidate universe")
    return ranking.sort_values(["qid", "rank"], kind="stable")


def freeze_jobs(
    *, rankings_path: Path, questions_path: Path, texts_path: Path, candidate_order_path: Path,
    prompt_path: Path, output_path: Path, family: str, modality: str, model_id: str, model_revision: str,
) -> dict[str, Any]:
    """Write a new, immutable per-position E030 job list and its provenance receipt."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable E030 job list: {output_path}")
    if modality not in {"article", "jp"}:
        raise ValueError("modality must be article or jp")
    questions = _load_questions(questions_path)
    cards = _load_cards(texts_path, modality)
    allowed_ids = {str(value) for value in np.load(candidate_order_path, allow_pickle=True).tolist()}
    ranking = _select_and_validate_ranking(
        rankings_path, questions=questions, allowed_ids=allowed_ids, modality=modality,
    )
    source_hashes = {
        "source_ranking_sha256": sha256(rankings_path),
        "source_questions_sha256": sha256(questions_path),
        "source_cards_sha256": sha256(texts_path),
        "candidate_order_sha256": sha256(candidate_order_path),
        "prompt_sha256": sha256(prompt_path),
    }
    jobs: list[dict[str, Any]] = []
    for row in ranking.to_dict("records"):
        candidate_id = str(row["item_id"])
        card = cards.get(candidate_id)
        if card is None:
            raise ValueError(f"{row['qid']}: candidate has no visible {modality} card")
        job = {
            "experiment_id": "E030",
            "family": family,
            "modality": modality,
            "qid": str(row["qid"]),
            "position": int(row["rank"]),
            "question": questions[str(row["qid"])],
            "candidate_id_internal": candidate_id,
            "document": card,
            "model_id": model_id,
            "model_revision": model_revision,
            "temperature": 0,
            **source_hashes,
        }
        job["job_id"] = canonical_sha256(job)
        jobs.append(job)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(canonical_json(job) + "\n" for job in jobs), encoding="utf-8")
    receipt = {
        "schema_version": "b2-e030-judge-jobs.v1",
        "experiment_id": "E030",
        "family": family,
        "modality": modality,
        "questions": len(questions),
        "jobs": len(jobs),
        "jobs_sha256": sha256(output_path),
        "model_id": model_id,
        "model_revision": model_revision,
        **source_hashes,
    }
    receipt_path = output_path.with_suffix(output_path.suffix + ".receipt.json")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable E030 receipt: {receipt_path}")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--cards", type=Path, required=True)
    parser.add_argument("--candidate-order", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--modality", choices=("article", "jp"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = freeze_jobs(
        rankings_path=args.rankings, questions_path=args.questions, texts_path=args.cards,
        candidate_order_path=args.candidate_order, prompt_path=args.prompt, output_path=args.output,
        family=args.family, modality=args.modality, model_id=args.model_id, model_revision=args.model_revision,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
