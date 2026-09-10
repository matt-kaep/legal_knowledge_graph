#!/usr/bin/env python3
"""Materialize a training benchmark snapshot with no train/evaluation overlap.

The evaluation split is preserved byte-for-byte.  Any training question that
shares either a QID or a normalized question text with evaluation is excluded.
The source files are never modified; the destination must not already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(
    os.environ.get(
        "LKG_REPO",
        "/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph",
    )
)
BENCH_ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
SOURCE_TRAIN_SPLIT = "train_augmented_retrievable_strict"
EVALUATION_SPLIT = "eval_rich_retrievable_strict"
FROZEN_TRAIN_SPLIT = "train_augmented_retrievable_strict_no_eval_overlap_v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_question_text(text: object) -> str:
    """Match the grouped-fold normalization used by 41_make_kfold_assignments."""
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    alphanumeric = "".join(char if char.isalnum() else " " for char in normalized)
    return " ".join(alphanumeric.split())


def _question_qid(question: dict[str, Any]) -> str:
    qid = str(question.get("qid", "")).strip()
    if not qid:
        raise ValueError("every question must contain a non-empty qid")
    return qid


def _question_text(question: dict[str, Any]) -> str:
    text = normalize_question_text(question.get("enonce", ""))
    if not text:
        raise ValueError(f"question {_question_qid(question)} has no usable enonce")
    return text


def _gold_ids(question: dict[str, Any], field: str) -> list[str]:
    values = question.get(field)
    if not isinstance(values, list):
        raise ValueError(f"question {_question_qid(question)} has invalid {field}")
    normalized = [str(value) for value in values]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"question {_question_qid(question)} has duplicate {field}")
    return normalized


def _load_split(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    required = {
        "bench_global.json": directory / "bench_global.json",
        "questions_ids.npy": directory / "questions_ids.npy",
        "questions_emb.npy": directory / "questions_emb.npy",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing split files in {directory}: {missing}")
    payload = json.loads(required["bench_global.json"].read_text(encoding="utf-8"))
    questions = payload.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"{required['bench_global.json']} must contain a questions list")
    qids = [_question_qid(question) for question in questions]
    if len(qids) != len(set(qids)):
        raise ValueError(f"duplicate QIDs in {required['bench_global.json']}")
    for question in questions:
        _question_text(question)
        _gold_ids(question, "articles_attendus")
        _gold_ids(question, "gold_jp_ids")
    ids = np.load(required["questions_ids.npy"], allow_pickle=True)
    embeddings = np.load(required["questions_emb.npy"])
    if ids.ndim != 1:
        raise ValueError(f"{required['questions_ids.npy']} must be one-dimensional")
    if embeddings.ndim < 1:
        raise ValueError(f"{required['questions_emb.npy']} must have a question axis")
    cached_qids = [str(qid) for qid in ids.tolist()]
    if cached_qids != qids:
        raise ValueError(f"question ID cache does not match bench order in {directory}")
    if embeddings.shape[0] != len(questions):
        raise ValueError(f"embedding cache length does not match bench questions in {directory}")
    return payload, questions, ids, embeddings


def _same_annotations(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        _gold_ids(left, "articles_attendus") == _gold_ids(right, "articles_attendus")
        and _gold_ids(left, "gold_jp_ids") == _gold_ids(right, "gold_jp_ids")
    )


def _split_counts(questions: list[dict[str, Any]]) -> dict[str, int]:
    article_lists = [_gold_ids(question, "articles_attendus") for question in questions]
    decision_lists = [_gold_ids(question, "gold_jp_ids") for question in questions]
    return {
        "questions": len(questions),
        "unique_qids": len({_question_qid(question) for question in questions}),
        "distinct_annotated_articles": len({item for values in article_lists for item in values}),
        "distinct_annotated_decisions": len({item for values in decision_lists for item in values}),
        "question_article_links": sum(len(values) for values in article_lists),
        "question_decision_links": sum(len(values) for values in decision_lists),
        "max_article_references_per_question": max(map(len, article_lists), default=0),
        "max_decision_references_per_question": max(map(len, decision_lists), default=0),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_training_against_evaluation(
    *,
    train_dir: Path,
    evaluation_dir: Path,
    output_dir: Path,
    output_split: str,
) -> dict[str, Any]:
    """Create an immutable, filtered train snapshot and return its manifest."""
    train_dir = train_dir.resolve()
    evaluation_dir = evaluation_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir in {train_dir, evaluation_dir}:
        raise ValueError("output_dir must differ from both source split directories")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing frozen split: {output_dir}")

    train_payload, train_questions, train_ids, train_embeddings = _load_split(train_dir)
    _, evaluation_questions, _, _ = _load_split(evaluation_dir)
    evaluation_by_qid = {_question_qid(question): question for question in evaluation_questions}
    evaluation_texts = {_question_text(question) for question in evaluation_questions}

    exclusions: list[dict[str, Any]] = []
    kept_indices: list[int] = []
    frozen_questions: list[dict[str, Any]] = []
    for index, question in enumerate(train_questions):
        qid = _question_qid(question)
        normalized_text = _question_text(question)
        reasons: list[str] = []
        evaluation_match = evaluation_by_qid.get(qid)
        if evaluation_match is not None:
            if normalized_text != _question_text(evaluation_match):
                raise ValueError(f"shared qid has different normalized text: {qid}")
            if not _same_annotations(question, evaluation_match):
                raise ValueError(f"shared qid annotations differ: {qid}")
            reasons.append("qid")
        if normalized_text in evaluation_texts:
            reasons.append("normalized_text")
        if reasons:
            exclusions.append(
                {
                    "qid": qid,
                    "reasons": reasons,
                    "article_gold_count": len(_gold_ids(question, "articles_attendus")),
                    "decision_gold_count": len(_gold_ids(question, "gold_jp_ids")),
                }
            )
            continue
        frozen_question = dict(question)
        frozen_question["split"] = output_split
        frozen_questions.append(frozen_question)
        kept_indices.append(index)

    frozen_qids = {_question_qid(question) for question in frozen_questions}
    frozen_texts = {_question_text(question) for question in frozen_questions}
    evaluation_qids = set(evaluation_by_qid)
    if frozen_qids & evaluation_qids or frozen_texts & evaluation_texts:
        raise AssertionError("freeze did not eliminate every cross-split QID/text overlap")
    if not exclusions:
        raise ValueError("refusing to materialize a no-op freeze with no detected overlap")

    frozen_payload = dict(train_payload)
    frozen_payload["split"] = output_split
    frozen_payload["questions"] = frozen_questions
    frozen_payload["frozen_from"] = {
        "source_split": str(train_payload.get("split", SOURCE_TRAIN_SPLIT)),
        "source_bench_sha256": sha256_file(train_dir / "bench_global.json"),
        "evaluation_split": str(EVALUATION_SPLIT),
        "evaluation_bench_sha256": sha256_file(evaluation_dir / "bench_global.json"),
        "exclusion_policy": "remove_train_questions_matching_eval_qid_or_normalized_text",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_json(temp_dir / "bench_global.json", frozen_payload)
        np.save(temp_dir / "questions_ids.npy", train_ids[kept_indices])
        np.save(temp_dir / "questions_emb.npy", train_embeddings[kept_indices])
        stats = {
            "schema_version": "benchmark-freeze-stats.v1",
            "split": output_split,
            "train": _split_counts(frozen_questions),
            "evaluation_preserved": _split_counts(evaluation_questions),
        }
        _write_json(temp_dir / "stats.json", stats)
        output_files = {
            name: sha256_file(temp_dir / name)
            for name in ("bench_global.json", "questions_ids.npy", "questions_emb.npy", "stats.json")
        }
        manifest = {
            "schema_version": "benchmark-freeze-manifest.v1",
            "freeze_id": "train-no-eval-overlap-v1",
            "output_split": output_split,
            "policy": {
                "evaluation": "preserved_without_modification",
                "training": "remove_questions_matching_evaluation_by_qid_or_normalized_text",
                "qid_match_requires_identical_normalized_text_and_exact_annotations": True,
                "text_normalization": "NFKC + casefold + alphanumeric tokens + collapsed whitespace",
            },
            "source_files": {
                "training": {
                    name: sha256_file(train_dir / name)
                    for name in ("bench_global.json", "questions_ids.npy", "questions_emb.npy")
                },
                "evaluation": {
                    name: sha256_file(evaluation_dir / name)
                    for name in ("bench_global.json", "questions_ids.npy", "questions_emb.npy")
                },
            },
            "exclusions": sorted(exclusions, key=lambda item: item["qid"]),
            "counts": {
                "train_questions_before": len(train_questions),
                "train_questions_after": len(frozen_questions),
                "evaluation_questions_preserved": len(evaluation_questions),
                "cross_split_qid_before": len({_question_qid(question) for question in train_questions} & evaluation_qids),
                "cross_split_qid_after": len(frozen_qids & evaluation_qids),
                "cross_split_normalized_text_before": sum(
                    _question_text(question) in evaluation_texts for question in train_questions
                ),
                "cross_split_normalized_text_after": len(frozen_texts & evaluation_texts),
            },
            "output_files": output_files,
        }
        _write_json(temp_dir / "freeze_manifest.json", manifest)
        temp_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=BENCH_ROOT / SOURCE_TRAIN_SPLIT)
    parser.add_argument("--evaluation-dir", type=Path, default=BENCH_ROOT / EVALUATION_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / FROZEN_TRAIN_SPLIT)
    parser.add_argument("--output-split", default=FROZEN_TRAIN_SPLIT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = freeze_training_against_evaluation(
        train_dir=args.train_dir,
        evaluation_dir=args.evaluation_dir,
        output_dir=args.output_dir,
        output_split=args.output_split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
