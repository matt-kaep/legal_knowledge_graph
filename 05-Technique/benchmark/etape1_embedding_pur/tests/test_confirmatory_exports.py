import importlib.util
import hashlib
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "66_export_confirmatory_results.py"
spec = importlib.util.spec_from_file_location("confirmatory_exports", SCRIPT)
exports = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(exports)


def test_expected_coverage_is_fraction_and_any_hit_is_distinct():
    rankings = pd.DataFrame(
        [
            {"qid": "q1", "method": "PPR", "modality": "art", "rank": 1, "item_id": "a1"},
            {"qid": "q1", "method": "PPR", "modality": "art", "rank": 2, "item_id": "noise"},
            {"qid": "q1", "method": "PPR", "modality": "art", "rank": 3, "item_id": "a2"},
        ]
    )
    expected = {("q1", "art"): {"a1", "a2"}}

    rows = exports.expected_coverage_rows(rankings, expected, ks=[1, 3])

    assert rows.loc[rows["k"] == 1, "expected_coverage_at_k"].item() == pytest.approx(0.5)
    assert rows.loc[rows["k"] == 1, "any_expected_answer_at_k"].item() == 1.0
    assert rows.loc[rows["k"] == 3, "expected_coverage_at_k"].item() == 1.0


def test_expected_coverage_keeps_distinct_target_specific_runs():
    rankings = pd.DataFrame([
        {"qid": "q1", "run_id": "art-run", "selection_target": "art", "method": "LightGCN", "modality": "art", "rank": 1, "item_id": "a1"},
        {"qid": "q1", "run_id": "jp-run", "selection_target": "jp", "method": "LightGCN", "modality": "art", "rank": 1, "item_id": "noise"},
    ])

    rows = exports.expected_coverage_rows(rankings, {("q1", "art"): {"a1"}}, ks=[1])

    assert len(rows) == 2
    assert set(rows["run_id"]) == {"art-run", "jp-run"}

    summary = exports.summarize_expected_coverage(rows)
    assert len(summary) == 2
    assert set(summary["run_id"]) == {"art-run", "jp-run"}


def test_paper_rows_require_provenance_and_scientific_status():
    rows = pd.DataFrame(
        [
            {
                "family": "ppr",
                "graph_version": "G1",
                "target": "articles_strict",
                "protocol_version": "grouped_v2",
                "dataset_sha256": "dataset",
                "fold_assignment_sha256": "folds",
                "source_artifact": "/tmp/source.csv",
                "experiment_id": "E002",
                "scientific_status": "confirmee_interne",
                "manifest_sha256": "manifest",
                "internal_eval_sha256": "eval",
                "graph_matrix_sha256": "graph",
                "eligible_champion": True,
                "question_coverage": 1.0,
            }
        ]
    )

    exports.validate_paper_rows(rows)
    with pytest.raises(ValueError, match="experiment_id"):
        exports.validate_paper_rows(rows.drop(columns="experiment_id"))


def test_primary_metric_plot_uses_recall_for_articles_and_hit_for_jp():
    rows = pd.DataFrame([
        {"graph_version": "G1", "family": "ppr", "target": "articles_strict", "m1": 0.4, "hit": 0.9},
        {"graph_version": "G1", "family": "ppr", "target": "jp", "m1": 0.8, "hit": 0.3},
    ])

    plotted = exports.primary_metric_plot_rows(rows)

    assert plotted["primary_metric"].tolist() == [0.4, 0.3]


def test_only_explicit_registry_classifications_authorize_paper_export(tmp_path):
    registry = tmp_path / "registry.csv"
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("proof\n1\n")
    registry.write_text(
        "experiment_id,statut,artefact_principal\n"
        "E002,confirmatoire_en_cours,\n"
        f"E003,refutee,{evidence}\n"
        f"E014,confirmee_interne,{evidence}\n"
    )

    statuses = exports.load_authorized_experiment_statuses(registry)

    assert statuses == {"E003": "refutee", "E014": "confirmee_interne"}


def test_result_verdicts_are_keyed_by_experiment_graph_family_and_target(tmp_path):
    registry = tmp_path / "results.csv"
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("proof\n1\n")
    registry.write_text(
        "result_id,experiment_id,graph_version,family,target,verdict,source_artifact,source_sha256\n"
        f"R1,E003,G1,lightgcn,articles_strict,confirmee_interne,{evidence},{hashlib.sha256(evidence.read_bytes()).hexdigest()}\n"
        f"R2,E003,G1,lightgcn,jp,refutee,{evidence},{hashlib.sha256(evidence.read_bytes()).hexdigest()}\n"
    )

    verdicts = exports.load_authorized_result_verdicts(registry)

    assert verdicts[("E003", "G1", "lightgcn", "articles_strict")] == "confirmee_interne"
    assert verdicts[("E003", "G1", "lightgcn", "jp")] == "refutee"


def test_result_verdict_rejects_changed_evidence(tmp_path):
    registry = tmp_path / "results.csv"
    evidence = tmp_path / "evidence.csv"
    evidence.write_text("changed\n")
    registry.write_text(
        "result_id,experiment_id,graph_version,family,target,verdict,source_artifact,source_sha256\n"
        f"R1,E003,G1,lightgcn,jp,refutee,{evidence},{'0' * 64}\n"
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        exports.load_authorized_result_verdicts(registry)
