from pathlib import Path
import hashlib
import importlib.util
import json

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "91_freeze_no_overlap_benchmark.py"
)
spec = importlib.util.spec_from_file_location("freeze_no_overlap", SCRIPT)
freeze_no_overlap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze_no_overlap)


def _question(qid: str, text: str, *, article: str = "article:1", decision: str = "jp:1") -> dict:
    return {
        "qid": qid,
        "enonce": text,
        "source": "fixture",
        "doc_id": "document",
        "section_id": "section",
        "split": "legacy",
        "articles_attendus": [article],
        "articles_attendus_etendu": [article],
        "gold_jp_ids": [decision],
        "n_articles_strict": 1,
        "n_articles_etendu": 1,
        "n_jp_resolues": 1,
    }


def _write_split(directory: Path, questions: list[dict]) -> None:
    directory.mkdir(parents=True)
    (directory / "bench_global.json").write_text(
        json.dumps(
            {
                "schema_version": "fixture.v1",
                "split": "legacy",
                "k": 10,
                "policy": "retrievable_strict_only",
                "questions": questions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(directory / "questions_ids.npy", np.array([q["qid"] for q in questions], dtype=object))
    np.save(
        directory / "questions_emb.npy",
        np.arange(len(questions) * 2, dtype=np.float32).reshape(len(questions), 2),
    )


def test_freeze_removes_train_overlap_preserves_eval_and_records_hashes(tmp_path: Path) -> None:
    train = tmp_path / "train"
    evaluation = tmp_path / "evaluation"
    output = tmp_path / "frozen-train"
    shared = _question("shared", "Texte identique", article="article:shared", decision="jp:shared")
    _write_split(train, [_question("keep", "Texte propre"), shared])
    _write_split(evaluation, [shared])
    train_before = (train / "bench_global.json").read_bytes()
    eval_before = (evaluation / "bench_global.json").read_bytes()

    manifest = freeze_no_overlap.freeze_training_against_evaluation(
        train_dir=train,
        evaluation_dir=evaluation,
        output_dir=output,
        output_split="train_no_overlap_v1",
    )

    frozen_questions = json.loads((output / "bench_global.json").read_text(encoding="utf-8"))["questions"]
    assert [question["qid"] for question in frozen_questions] == ["keep"]
    assert frozen_questions[0]["split"] == "train_no_overlap_v1"
    assert np.load(output / "questions_ids.npy", allow_pickle=True).tolist() == ["keep"]
    assert np.load(output / "questions_emb.npy").tolist() == [[0.0, 1.0]]
    assert (train / "bench_global.json").read_bytes() == train_before
    assert (evaluation / "bench_global.json").read_bytes() == eval_before
    assert manifest["counts"]["train_questions_before"] == 2
    assert manifest["counts"]["train_questions_after"] == 1
    assert manifest["counts"]["cross_split_qid_after"] == 0
    assert manifest["exclusions"] == [
        {
            "qid": "shared",
            "reasons": ["qid", "normalized_text"],
            "article_gold_count": 1,
            "decision_gold_count": 1,
        }
    ]
    for relative_path, expected_hash in manifest["output_files"].items():
        assert hashlib.sha256((output / relative_path).read_bytes()).hexdigest() == expected_hash


def test_freeze_rejects_a_shared_qid_with_different_annotations(tmp_path: Path) -> None:
    train = tmp_path / "train"
    evaluation = tmp_path / "evaluation"
    _write_split(train, [_question("shared", "Texte identique", article="article:train")])
    _write_split(evaluation, [_question("shared", "Texte identique", article="article:eval")])

    with pytest.raises(ValueError, match="annotations differ"):
        freeze_no_overlap.freeze_training_against_evaluation(
            train_dir=train,
            evaluation_dir=evaluation,
            output_dir=tmp_path / "frozen-train",
            output_split="train_no_overlap_v1",
        )
