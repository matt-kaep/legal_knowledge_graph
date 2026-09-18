"""Tests for the immutable A3 effective-retrieval snapshot generator."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "93_freeze_effective_retrieval_universe.py"
spec = importlib.util.spec_from_file_location("freeze_effective_retrieval", SCRIPT)
freeze_effective_retrieval = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(freeze_effective_retrieval)


def _write_bench(directory: Path, *, qid: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    question = {
        "qid": qid,
        "articles_attendus": ["art-1"],
        "articles_attendus_etendu": ["art-1", "art-2", "art-outside"],
        "gold_jp_ids": ["jp-1"],
    }
    (directory / "bench_global.json").write_text(
        json.dumps({"questions": [question]}, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(directory / "questions_ids.npy", np.asarray([qid], dtype=object))
    np.save(directory / "questions_emb.npy", np.asarray([[1.0, 2.0]], dtype=np.float32))
    (directory / "stats.json").write_text("{}\n", encoding="utf-8")


def test_freeze_a3_copies_the_benchmark_and_records_effective_candidate_contract(tmp_path):
    train = tmp_path / "train-v2"
    evaluation = tmp_path / "eval"
    output = tmp_path / "train-v3"
    manifest = tmp_path / "a3-freeze.json"
    _write_bench(train, qid="train-1")
    _write_bench(evaluation, qid="eval-1")
    np.save(
        tmp_path / "article-order.npy",
        np.asarray(["art-1", "art-representation-only", "art-2"], dtype=object),
    )
    np.save(
        tmp_path / "jp-order.npy",
        np.asarray(["jp-1", "jp-1", "jp-representation-only", "jp-2"], dtype=object),
    )
    np.save(tmp_path / "graph-articles.npy", np.asarray(["art-1", "art-2", "art-aux"], dtype=object))
    np.save(tmp_path / "graph-jp.npy", np.asarray(["jp-1", "jp-1", "jp-2", "jp-aux"], dtype=object))

    assert freeze_effective_retrieval.main(
        [
            "--source-train-bench-dir", str(train),
            "--evaluation-bench-dir", str(evaluation),
            "--output-train-bench-dir", str(output),
            "--article-order-path", str(tmp_path / "article-order.npy"),
            "--jp-order-path", str(tmp_path / "jp-order.npy"),
            "--graph-article-ids-path", str(tmp_path / "graph-articles.npy"),
            "--graph-jp-ids-path", str(tmp_path / "graph-jp.npy"),
            "--manifest-output", str(manifest),
        ]
    ) == 0

    projection = json.loads(
        (output / "lightgcn_article_positive_projection.json").read_text(encoding="utf-8")
    )
    snapshot = json.loads(manifest.read_text(encoding="utf-8"))
    assert (output / "bench_global.json").read_bytes() == (train / "bench_global.json").read_bytes()
    assert projection["counts"] == {
        "questions": 1,
        "extended_label_occurrences": 3,
        "extended_labels_present": 2,
        "extended_labels_absent": 1,
        "questions_with_absent_extended_labels": 1,
        "questions_without_retrievable_positive": 0,
    }
    assert snapshot["retrieval_candidate_universe"]["articles"]["unique_ids"] == 2
    assert snapshot["retrieval_candidate_universe"]["jurisprudence"]["unique_ids"] == 2
    assert snapshot["auxiliary_non_returnable_nodes"]["articles"]["unique_ids"] == 1
    assert snapshot["auxiliary_non_returnable_nodes"]["jurisprudence"]["unique_ids"] == 1
    assert snapshot["strict_label_coverage"] == {
        "train": {"articles_absent": 0, "jurisprudence_absent": 0},
        "evaluation": {"articles_absent": 0, "jurisprudence_absent": 0},
    }


def test_freeze_a3_refuses_to_overwrite_an_existing_snapshot(tmp_path):
    output = tmp_path / "already-exists"
    output.mkdir()

    try:
        freeze_effective_retrieval.ensure_output_does_not_exist(output)
    except FileExistsError:
        pass
    else:  # pragma: no cover - protects the immutable-snapshot requirement
        raise AssertionError("existing A3 output must never be overwritten")
