#!/usr/bin/env python3
"""Materialize immutable real candidate pools for B2/E029 comparable reranking.

Each pool is derived from exactly one frozen top-100 ranking source, one
modality and one K_in. When a source contains several final replay seeds, the
seed is explicit and cannot be mixed silently. Candidate text is joined by the
identifier itself; a missing, conflicting or out-of-A3 candidate is a hard
preflight failure rather than a silent exclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE_REPO = Path(os.environ.get("LKG_REPO", str(ROOT.parents[3]))).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_unique(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        identifier = str(value)
        if identifier not in seen:
            result.append(identifier)
            seen.add(identifier)
    return result


def _data_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else DATA_REPO / path


def a3_candidate_ids(a3_payload: dict[str, Any], modality: str) -> set[str]:
    target = "articles" if modality == "article" else "jurisprudence"
    retrieval = a3_payload["candidate_universes"]["retrieval_candidate_universe"][target]
    structural = a3_payload["candidate_universes"]["graph_node_universe"][target]
    representation = np.load(_data_path(retrieval["representation_order_path"]), allow_pickle=True).tolist()
    graph = set(map(str, np.load(_data_path(structural["path"]), allow_pickle=True).tolist()))
    candidate_ids = stable_unique(value for value in representation if str(value) in graph)
    if len(candidate_ids) != int(retrieval["unique_ids"]):
        raise ValueError(f"A3 {target} candidate count differs from its manifest")
    return set(candidate_ids)


def load_questions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("questions file must contain a questions list")
    result: dict[str, str] = {}
    for question in questions:
        qid = str(question.get("qid"))
        text = str(question.get("enonce") or "").strip()
        if not qid or not text or qid in result:
            raise ValueError("questions require unique qid values and non-empty enonce values")
        result[qid] = text
    return result


def load_texts(path: Path, modality: str) -> dict[str, str]:
    frame = pd.read_parquet(path)
    id_column, text_column = ("pair_key", "texte") if modality == "article" else ("jp_id", "synthese")
    missing = {id_column, text_column} - set(frame.columns)
    if missing:
        raise ValueError(f"{modality} text source missing columns: {sorted(missing)}")
    frame = frame[[id_column, text_column]].copy()
    frame[id_column] = frame[id_column].astype(str)
    frame[text_column] = frame[text_column].fillna("").astype(str)
    if (frame[text_column].str.strip() == "").any():
        raise ValueError(f"{modality} text source contains empty candidate text")
    conflicts = frame.groupby(id_column)[text_column].nunique()
    if (conflicts > 1).any():
        raise ValueError(f"{modality} text source contains conflicting text for one identifier")
    frame = frame.drop_duplicates(subset=[id_column], keep="first")
    return dict(zip(frame[id_column], frame[text_column], strict=True))


def materialize_pool(
    *, ranking_path: Path, questions_path: Path, text_source_path: Path, output_path: Path,
    family: str, modality: str, candidate_ids: set[str], k_in: int, method: str | None = None,
    replay_seed: str | None = None,
) -> int:
    if modality not in {"article", "jp"}:
        raise ValueError("modality must be article or jp")
    if k_in <= 0 or k_in > 100:
        raise ValueError("K_in must be in 1..100")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable pool: {output_path}")
    questions = load_questions(questions_path)
    ranking = pd.read_parquet(ranking_path)
    required = {"qid", "method", "modality", "rank", "item_id"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"ranking missing columns: {sorted(missing)}")
    source_modality = "art" if modality == "article" else "jp"
    ranking = ranking.loc[ranking["modality"].astype(str).eq(source_modality)].copy()
    if "selected_target" in ranking.columns:
        expected_target = "art" if modality == "article" else "jp"
        ranking = ranking.loc[ranking["selected_target"].astype(str).eq(expected_target)].copy()
    methods = sorted(ranking["method"].astype(str).unique())
    if method is None:
        if len(methods) != 1:
            raise ValueError(f"frozen ranking requires explicit method; found {methods}")
        method = methods[0]
    ranking = ranking.loc[ranking["method"].astype(str).eq(method)].copy()
    seed_column = "replay_seed" if "replay_seed" in ranking.columns else "seed" if "seed" in ranking.columns else None
    if replay_seed is not None:
        if seed_column is None:
            raise ValueError("frozen ranking has no replay-seed column")
        ranking = ranking.loc[ranking[seed_column].astype(str).eq(str(replay_seed))].copy()
        if ranking.empty:
            raise ValueError(f"frozen ranking has no rows for replay seed {replay_seed}")
    elif seed_column is not None and ranking[seed_column].nunique() > 1:
        raise ValueError("frozen ranking contains several replay seeds; --replay-seed is required")
    ranking["qid"] = ranking["qid"].astype(str)
    ranking["item_id"] = ranking["item_id"].astype(str)
    ranking["rank"] = ranking["rank"].astype(int)
    if set(ranking["qid"]) != set(questions):
        raise ValueError("ranking question coverage differs from the frozen question set")
    texts = load_texts(text_source_path, modality)
    ranking_sha = sha256(ranking_path)
    texts_sha = sha256(text_source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("x", encoding="utf-8") as output:
        for qid, group in ranking.groupby("qid", sort=True):
            group = group.sort_values("rank", kind="stable")
            ranks = group["rank"].tolist()
            if ranks[:k_in] != list(range(1, k_in + 1)):
                raise ValueError(f"{family}/{modality}/{qid}: expected frozen ranks 1..K_in")
            ids = group["item_id"].tolist()
            selected_ids = ids[:k_in]
            if len(selected_ids) != len(set(selected_ids)):
                raise ValueError(f"{family}/{modality}/{qid}: duplicate identifier in frozen top-K_in")
            outside = set(selected_ids) - candidate_ids
            if outside:
                raise ValueError(f"{family}/{modality}/{qid}: candidate outside A3 universe: {sorted(outside)[:3]}")
            candidates: list[dict[str, Any]] = []
            for rank, item_id in enumerate(selected_ids, start=1):
                text = texts.get(item_id)
                if text is None:
                    raise ValueError(f"{family}/{modality}/{qid}: missing candidate text for {item_id}")
                candidates.append({"item_id": item_id, "text": text, "source_rank": rank})
            record = {
                "experiment_id": "E029",
                "family": family,
                "modality": modality,
                "qid": qid,
                "question": questions[qid],
                "k_in": k_in,
                "k_out": 10,
                "source_method": method,
                "source_ranking_sha256": ranking_sha,
                "source_texts_sha256": texts_sha,
                "candidates": candidates,
            }
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=("cosine", "ppr", "lightgcn"), required=True)
    parser.add_argument("--modality", choices=("article", "jp"), required=True)
    parser.add_argument("--a3-manifest", type=Path, required=True)
    parser.add_argument("--method")
    parser.add_argument("--replay-seed")
    parser.add_argument("--k-in", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    a3_payload = json.loads(args.a3_manifest.read_text(encoding="utf-8"))
    count = materialize_pool(
        ranking_path=args.ranking,
        questions_path=args.questions,
        text_source_path=args.texts,
        output_path=args.output,
        family=args.family,
        modality=args.modality,
        candidate_ids=a3_candidate_ids(a3_payload, args.modality),
        k_in=args.k_in,
        method=args.method,
        replay_seed=args.replay_seed,
    )
    print(json.dumps({"questions": count, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
