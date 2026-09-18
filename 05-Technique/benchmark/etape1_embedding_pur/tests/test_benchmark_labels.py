from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_labels


def _question(
    qid: str,
    *,
    strict_articles: list[str],
    extended_articles: list[str],
    strict_jp: list[str],
) -> dict:
    return {
        "qid": qid,
        "articles_attendus": strict_articles,
        "articles_attendus_etendu": extended_articles,
        "gold_jp_ids": strict_jp,
    }


def test_strict_labels_absent_from_candidates_are_reported_not_discarded():
    questions = [
        _question(
            "q-ok",
            strict_articles=["article:1"],
            extended_articles=["article:1"],
            strict_jp=["jp:1"],
        ),
        _question(
            "q-invalid",
            strict_articles=["article:missing"],
            extended_articles=["article:missing"],
            strict_jp=["jp:missing"],
        ),
    ]
    issues = benchmark_labels.strict_candidate_coverage_issues(
        questions,
        article_candidate_ids=["article:1"],
        jp_candidate_ids=["jp:1"],
    )

    assert issues == [
        {
            "qid": "q-invalid",
            "missing_articles_attendus": ["article:missing"],
            "missing_gold_jp_ids": ["jp:missing"],
        }
    ]
    with pytest.raises(ValueError, match="strict labels absent from candidate spaces"):
        benchmark_labels.require_strict_candidate_coverage(
            questions,
            article_candidate_ids=["article:1"],
            jp_candidate_ids=["jp:1"],
            context="fixture",
        )


def test_lightgcn_projection_records_extended_labels_excluded_from_training(tmp_path: Path):
    questions = [
        _question(
            "q1",
            strict_articles=["article:1"],
            extended_articles=["article:1", "article:missing", "article:1"],
            strict_jp=["jp:1"],
        )
    ]
    bench_path = tmp_path / "bench_global.json"
    bench_path.write_text(json.dumps({"questions": questions}), encoding="utf-8")

    projection = benchmark_labels.build_lightgcn_article_positive_projection(
        questions,
        article_candidate_ids=["article:1", "article:2"],
        bench_sha256=benchmark_labels.sha256_file(bench_path),
    )

    assert projection["counts"] == {
        "questions": 1,
        "extended_label_occurrences": 2,
        "extended_labels_present": 1,
        "extended_labels_absent": 1,
        "questions_with_absent_extended_labels": 1,
        "questions_without_retrievable_positive": 0,
    }
    assert projection["rows"] == [
        {
            "qid": "q1",
            "extended_article_ids": ["article:1", "article:missing"],
            "retrievable_positive_article_ids": ["article:1"],
            "excluded_extended_article_ids": ["article:missing"],
        }
    ]


def test_ranking_validation_rejects_a_candidate_outside_the_official_universe():
    with pytest.raises(ValueError, match="outside the official retrieval candidate universe"):
        benchmark_labels.require_ranked_ids_within_candidate_universe(
            ["art-1", "art-outside"],
            candidate_ids=["art-1", "art-2"],
            context="test ranking",
        )


def test_ranking_validation_rejects_duplicate_candidate_after_first_occurrence():
    with pytest.raises(ValueError, match="duplicate candidate"):
        benchmark_labels.require_ranked_ids_within_candidate_universe(
            ["jp-1", "jp-1"],
            candidate_ids=["jp-1", "jp-2"],
            context="test ranking",
        )
