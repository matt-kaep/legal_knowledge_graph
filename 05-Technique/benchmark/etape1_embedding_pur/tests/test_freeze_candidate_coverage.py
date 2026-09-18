from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "92_freeze_train_candidate_coverage.py"
)
spec = importlib.util.spec_from_file_location("freeze_candidate_coverage", SCRIPT)
freeze_candidate_coverage = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(freeze_candidate_coverage)


def test_default_candidate_paths_are_resolved_inside_the_benchmark_tree():
    assert str(freeze_candidate_coverage.DEFAULT_ARTICLE_IDS).endswith(
        "05-Technique/benchmark/etape1_embedding_pur/data/hybrid_graphs/"
        "G6-citation-AA-knn5/article_ids.npy"
    )
    assert str(freeze_candidate_coverage.DEFAULT_JP_IDS).endswith(
        "05-Technique/benchmark/etape1_embedding_pur/data/hybrid_graphs/"
        "G6-citation-AA-knn5/jp_ids.npy"
    )


def _question(qid: str, article: str, extended: list[str], decision: str) -> dict:
    return {
        "qid": qid,
        "enonce": f"Question {qid}",
        "split": "train-no-eval-v1",
        "articles_attendus": [article],
        "articles_attendus_etendu": extended,
        "gold_jp_ids": [decision],
        "n_articles_strict": 1,
        "n_articles_etendu": len(extended),
        "n_jp_resolues": 1,
    }


def _write_split(directory: Path, questions: list[dict]) -> None:
    directory.mkdir(parents=True)
    (directory / "bench_global.json").write_text(
        json.dumps({"split": "train-no-eval-v1", "questions": questions}, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(directory / "questions_ids.npy", np.array([q["qid"] for q in questions], dtype=object))
    np.save(directory / "questions_emb.npy", np.arange(len(questions) * 2, dtype=np.float32).reshape(len(questions), 2))


def test_freeze_removes_every_strict_candidate_coverage_failure_and_writes_projection(tmp_path: Path):
    train = tmp_path / "train-v1"
    evaluation = tmp_path / "evaluation"
    output = tmp_path / "train-v2"
    keep = _question("keep", "article:1", ["article:1", "article:extended-missing"], "jp:1")
    missing_article = _question("missing-article", "article:missing", ["article:missing"], "jp:1")
    missing_jp = _question("missing-jp", "article:1", ["article:1"], "jp:missing")
    _write_split(train, [keep, missing_article, missing_jp])
    _write_split(evaluation, [_question("eval", "article:1", ["article:1"], "jp:1")])
    article_ids = tmp_path / "article_ids.npy"
    jp_ids = tmp_path / "jp_ids.npy"
    np.save(article_ids, np.array(["article:1"], dtype=str))
    np.save(jp_ids, np.array(["jp:1", "jp:1"], dtype=str))
    evaluation_before = (evaluation / "bench_global.json").read_bytes()

    manifest = freeze_candidate_coverage.freeze_training_to_candidate_coverage(
        train_dir=train,
        evaluation_dir=evaluation,
        article_ids_path=article_ids,
        jp_ids_path=jp_ids,
        output_dir=output,
        output_split="train-no-eval-candidate-v2",
    )

    frozen = json.loads((output / "bench_global.json").read_text(encoding="utf-8"))
    assert [question["qid"] for question in frozen["questions"]] == ["keep"]
    assert (evaluation / "bench_global.json").read_bytes() == evaluation_before
    assert manifest["counts"]["train_questions_before"] == 3
    assert manifest["counts"]["train_questions_after"] == 1
    assert manifest["counts"]["strict_candidate_coverage_failures_before"] == 2
    assert manifest["counts"]["strict_candidate_coverage_failures_after"] == 0
    assert manifest["counts"]["cross_split_qid_after"] == 0
    assert manifest["counts"]["cross_split_normalized_text_after"] == 0
    assert manifest["candidate_spaces"]["articles"]["unique_ids"] == 1
    assert manifest["candidate_spaces"]["jurisprudence"]["unique_ids"] == 1
    projection = json.loads((output / "lightgcn_article_positive_projection.json").read_text(encoding="utf-8"))
    assert projection["counts"]["extended_labels_present"] == 1
    assert projection["counts"]["extended_labels_absent"] == 1
    assert projection["rows"][0]["retrievable_positive_article_ids"] == ["article:1"]
    for relative_path, expected_hash in manifest["output_files"].items():
        assert hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == expected_hash
