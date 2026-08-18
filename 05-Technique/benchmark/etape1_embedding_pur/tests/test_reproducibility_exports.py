from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "90_audit_and_export_reproducibility.py"
SPEC = importlib.util.spec_from_file_location("repro_exports", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_classify_evidence_states_keeps_missing_and_exploratory_distinct():
    assert MODULE.classify_evidence(exists=True, complete=True, scientific_status="complete") == "complete"
    assert MODULE.classify_evidence(exists=True, complete=False, scientific_status="complete") == "incomplete"
    assert MODULE.classify_evidence(exists=False, complete=False, scientific_status="complete") == "missing"
    assert MODULE.classify_evidence(exists=True, complete=True, scientific_status="exploratory") == "exploratory"
    assert MODULE.classify_evidence(exists=True, complete=False, scientific_status="exploratory") == "incomplete"


def test_portable_relative_path_never_emits_a_personal_checkout(tmp_path):
    artifact = tmp_path / "data" / "artifact.csv"
    assert MODULE.portable_relative_path(artifact, tmp_path) == "data/artifact.csv"


def test_optional_sha256_returns_none_for_missing_artifact(tmp_path):
    assert MODULE.optional_sha256(tmp_path / "missing.json") is None


def test_e017_replay_coverage_requires_all_questions_and_ranks():
    complete = pd.DataFrame(
        [(qid, rank) for qid in range(754) for rank in range(1, 11)],
        columns=["qid", "rank"],
    )
    incomplete = complete.iloc[:-1]

    assert MODULE._ranking_coverage(complete, questions=754, k=10)
    assert not MODULE._ranking_coverage(incomplete, questions=754, k=10)


def test_e016_unique_judged_pairs_prefers_source_summary_count():
    assert MODULE._unique_judged_pairs({"n_unique_judged_pairs": 7487}, 7540) == 7487
    assert MODULE._unique_judged_pairs({"n_unique_jobs": 7487}, 7540) == 7487
    assert MODULE._unique_judged_pairs({}, 7540) == 7540
