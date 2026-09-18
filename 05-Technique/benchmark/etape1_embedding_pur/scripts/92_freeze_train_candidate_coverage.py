"""Freeze a new train snapshot after enforcing strict candidate coverage."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(os.environ.get("LKG_REPO", str(Path(__file__).resolve().parents[4]))).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(REPO))).expanduser().resolve()
ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_labels


BENCH_ROOT = DATA_REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
SOURCE_TRAIN_SPLIT = "train_augmented_retrievable_strict_no_eval_overlap_v1"
OUTPUT_TRAIN_SPLIT = "train_augmented_retrievable_strict_no_eval_overlap_candidate_covered_v2"
EVALUATION_SPLIT = "eval_rich_retrievable_strict"
DEFAULT_ARTICLE_IDS = (
    DATA_REPO
    / "05-Technique/benchmark/etape1_embedding_pur/data/hybrid_graphs/"
    "G6-citation-AA-knn5/article_ids.npy"
)
DEFAULT_JP_IDS = (
    DATA_REPO
    / "05-Technique/benchmark/etape1_embedding_pur/data/hybrid_graphs/"
    "G6-citation-AA-knn5/jp_ids.npy"
)


def normalize_question_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    alphanumeric = "".join(char if char.isalnum() else " " for char in normalized)
    return " ".join(alphanumeric.split())


def _question_text(question: dict[str, Any]) -> str:
    for key in ("enonce", "question"):
        value = question.get(key)
        if value is not None and str(value).strip():
            return normalize_question_text(value)
    return ""


def _load_split(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray, np.ndarray]:
    payload = json.loads((directory / "bench_global.json").read_text(encoding="utf-8"))
    questions = list(payload["questions"])
    qids = np.load(directory / "questions_ids.npy", allow_pickle=True)
    embeddings = np.load(directory / "questions_emb.npy")
    question_qids = [str(question["qid"]) for question in questions]
    array_qids = [str(qid) for qid in qids.tolist()]
    if len(question_qids) != len(set(question_qids)):
        raise ValueError(f"duplicate QIDs in benchmark questions: {directory}")
    if len(array_qids) != len(set(array_qids)):
        raise ValueError(f"duplicate QIDs in embedding cache: {directory}")
    if set(question_qids) != set(array_qids) or len(qids) != len(embeddings):
        raise ValueError(f"unaligned question/embedding artifacts: {directory}")
    return payload, questions, qids, embeddings


def _candidate_space(path: Path) -> dict[str, Any]:
    raw = np.load(path, allow_pickle=True).tolist()
    unique = benchmark_labels.stable_unique_strings(raw)
    return {
        "source_path": str(path),
        "source_sha256": benchmark_labels.sha256_file(path),
        "raw_rows": len(raw),
        "unique_ids": len(unique),
        "duplicate_rows": len(raw) - len(unique),
        "stable_unique_sequence_sha256": benchmark_labels.stable_sequence_sha256(unique),
        "ids": unique,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_training_to_candidate_coverage(
    *,
    train_dir: Path,
    evaluation_dir: Path,
    article_ids_path: Path,
    jp_ids_path: Path,
    output_dir: Path,
    output_split: str,
) -> dict[str, Any]:
    train_dir = train_dir.resolve()
    evaluation_dir = evaluation_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir in {train_dir, evaluation_dir}:
        raise ValueError("output_dir must differ from source train and evaluation directories")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen split: {output_dir}")

    train_payload, train_questions, train_qids, train_embeddings = _load_split(train_dir)
    evaluation_payload = json.loads((evaluation_dir / "bench_global.json").read_text(encoding="utf-8"))
    evaluation_questions = list(evaluation_payload["questions"])
    article_space = _candidate_space(article_ids_path)
    jp_space = _candidate_space(jp_ids_path)

    evaluation_qids = {str(question["qid"]) for question in evaluation_questions}
    evaluation_texts = {_question_text(question) for question in evaluation_questions}
    source_qids = {str(question["qid"]) for question in train_questions}
    source_texts = {_question_text(question) for question in train_questions}
    if source_qids & evaluation_qids or source_texts & evaluation_texts:
        raise ValueError("source train snapshot still overlaps evaluation by QID or normalized text")

    exclusions = benchmark_labels.strict_candidate_coverage_issues(
        train_questions,
        article_candidate_ids=article_space["ids"],
        jp_candidate_ids=jp_space["ids"],
    )
    excluded_qids = {str(row["qid"]) for row in exclusions}
    frozen_questions = [
        dict(question, split=output_split)
        for question in train_questions
        if str(question["qid"]) not in excluded_qids
    ]
    keep_mask = np.array([str(qid) not in excluded_qids for qid in train_qids.tolist()], dtype=bool)
    frozen_qids_array = train_qids[keep_mask]
    frozen_embeddings = train_embeddings[keep_mask]
    frozen_qids = {str(question["qid"]) for question in frozen_questions}
    frozen_texts = {_question_text(question) for question in frozen_questions}
    if len(frozen_questions) != len(frozen_qids_array):
        raise AssertionError("filtered question and embedding cache lengths differ")
    if frozen_qids & evaluation_qids or frozen_texts & evaluation_texts:
        raise AssertionError("candidate-coverage freeze reintroduced train/eval overlap")
    remaining_issues = benchmark_labels.strict_candidate_coverage_issues(
        frozen_questions,
        article_candidate_ids=article_space["ids"],
        jp_candidate_ids=jp_space["ids"],
    )
    if remaining_issues:
        raise AssertionError(f"strict candidate coverage remains incomplete: {remaining_issues[:5]}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        frozen_payload = dict(train_payload)
        frozen_payload["split"] = output_split
        frozen_payload["questions"] = frozen_questions
        frozen_payload["frozen_from"] = {
            "source_split": train_payload.get("split", SOURCE_TRAIN_SPLIT),
            "source_bench_sha256": benchmark_labels.sha256_file(train_dir / "bench_global.json"),
            "candidate_coverage_policy": "remove training questions with an absent strict Article or JP label",
            "evaluation_split_verified_unchanged": EVALUATION_SPLIT,
            "evaluation_bench_sha256": benchmark_labels.sha256_file(evaluation_dir / "bench_global.json"),
        }
        _write_json(temp_dir / "bench_global.json", frozen_payload)
        np.save(temp_dir / "questions_ids.npy", frozen_qids_array)
        np.save(temp_dir / "questions_emb.npy", frozen_embeddings)
        projection = benchmark_labels.write_lightgcn_article_positive_projection(
            temp_dir,
            frozen_questions,
            article_candidate_ids=article_space["ids"],
        )
        stats = {
            "schema_version": "benchmark-freeze-stats.v2",
            "split": output_split,
            "train_questions": len(frozen_questions),
            "evaluation_questions_preserved": len(evaluation_questions),
            "strict_candidate_coverage_failures": 0,
            "lightgcn_extended_article_projection": projection["counts"],
        }
        _write_json(temp_dir / "stats.json", stats)
        output_files = {
            name: benchmark_labels.sha256_file(temp_dir / name)
            for name in (
                "bench_global.json",
                "questions_ids.npy",
                "questions_emb.npy",
                "lightgcn_article_positive_projection.json",
                "stats.json",
            )
        }
        manifest = {
            "schema_version": "benchmark-freeze-manifest.v2",
            "freeze_id": "train-no-eval-overlap-candidate-coverage-v2",
            "output_split": output_split,
            "source": {
                "train_split": train_payload.get("split", SOURCE_TRAIN_SPLIT),
                "train_bench_sha256": benchmark_labels.sha256_file(train_dir / "bench_global.json"),
                "evaluation_split": EVALUATION_SPLIT,
                "evaluation_bench_sha256": benchmark_labels.sha256_file(evaluation_dir / "bench_global.json"),
            },
            "candidate_spaces": {
                "articles": {key: value for key, value in article_space.items() if key != "ids"},
                "jurisprudence": {key: value for key, value in jp_space.items() if key != "ids"},
            },
            "strict_candidate_coverage_exclusions": exclusions,
            "counts": {
                "train_questions_before": len(train_questions),
                "train_questions_after": len(frozen_questions),
                "evaluation_questions_preserved": len(evaluation_questions),
                "strict_candidate_coverage_failures_before": len(exclusions),
                "strict_candidate_coverage_failures_after": len(remaining_issues),
                "cross_split_qid_after": len(frozen_qids & evaluation_qids),
                "cross_split_normalized_text_after": len(frozen_texts & evaluation_texts),
            },
            "lightgcn_extended_article_projection": {
                "path": benchmark_labels.LIGHTGCN_PROJECTION_FILENAME,
                "sha256": output_files[benchmark_labels.LIGHTGCN_PROJECTION_FILENAME],
                "counts": projection["counts"],
                "candidate_sequence_sha256": projection["article_candidate_sequence_sha256"],
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
    parser.add_argument("--article-ids", type=Path, default=DEFAULT_ARTICLE_IDS)
    parser.add_argument("--jp-ids", type=Path, default=DEFAULT_JP_IDS)
    parser.add_argument("--output-dir", type=Path, default=BENCH_ROOT / OUTPUT_TRAIN_SPLIT)
    parser.add_argument("--output-split", default=OUTPUT_TRAIN_SPLIT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = freeze_training_to_candidate_coverage(
        train_dir=args.train_dir,
        evaluation_dir=args.evaluation_dir,
        article_ids_path=args.article_ids,
        jp_ids_path=args.jp_ids,
        output_dir=args.output_dir,
        output_split=args.output_split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
