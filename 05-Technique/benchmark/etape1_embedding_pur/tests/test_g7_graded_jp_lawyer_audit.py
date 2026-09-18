import importlib.util
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SELECT = load_module("g7_lawyer_select", "78_select_g7_graded_jp_lawyer_audit.py")
SUMMARY = load_module("g7_lawyer_summary", "79_summarize_g7_graded_jp_lawyer_audit.py")


def population(per_label=30):
    rows = []
    rank = 0
    for label in ("A", "B", "C", "D", "E", "non_jugeable"):
        for index in range(per_label):
            rank = index % 10 + 1
            rows.append(
                {
                    "qid": f"q-{label}-{index}",
                    "jp_id": f"jp-{label}-{index}",
                    "job_id": f"job-{label}-{index}",
                    "rank": rank,
                    "classe": label,
                    "justification": f"Justification {label} {index}",
                    "exact_gold": index % 3 == 0,
                }
            )
    return pd.DataFrame(rows)


def test_select_sample_uses_target_allocation_and_seed():
    first = SELECT.select_sample(population(), sample_size=100, seed=42)
    second = SELECT.select_sample(population().sample(frac=1, random_state=7), sample_size=100, seed=42)
    assert first["job_id"].tolist() == second["job_id"].tolist()
    assert first["classe"].value_counts().to_dict() == {
        "A": 25,
        "B": 20,
        "C": 15,
        "D": 15,
        "E": 15,
        "non_jugeable": 10,
    }
    assert (first["sampling_weight"] >= 1).all()


def test_blind_export_omits_llm_label_rank_gt_and_g8():
    selected = SELECT.select_sample(population(), sample_size=100, seed=42)
    questions = {
        qid: {"enonce": f"Question pour {qid}", "gold_jp_ids": ["secret-gold"]}
        for qid in selected["qid"]
    }
    cards = {jp_id: {"solution_resume": f"Solution {jp_id}"} for jp_id in selected["jp_id"]}
    blind, private = SELECT.build_exports(selected, questions=questions, cards=cards)
    forbidden = {"classe", "justification", "rank", "exact_gold", "gold_jp_ids", "sampling_weight"}
    assert not (forbidden & set(blind.columns))
    assert {"classe_avocat", "justification_avocat"}.issubset(blind.columns)
    assert blind["classe_avocat"].eq("").all()
    assert "secret-gold" not in blind.to_json(force_ascii=False)
    assert {"classe", "rank", "sampling_weight", "case_id"}.issubset(private.columns)


def test_agreement_reweights_stratified_sample_to_population():
    annotations = pd.DataFrame(
        [
            {"case_id": "c1", "classe_avocat": "A", "justification_avocat": "Règle A."},
            {"case_id": "c2", "classe_avocat": "E", "justification_avocat": "Aucun lien."},
        ]
    )
    key = pd.DataFrame(
        [
            {"case_id": "c1", "classe": "A", "sampling_weight": 9.0},
            {"case_id": "c2", "classe": "A", "sampling_weight": 1.0},
        ]
    )
    report = SUMMARY.summarize_agreement(annotations, key, expected_count=2, n_bootstrap=50)
    assert report["gain_agreement"] == 0.9
    assert report["mean_absolute_gain_error"] == 0.1
    assert report["positive_precision"] == 0.9


def test_gate_requires_gain_agreement_and_positive_precision():
    annotations = pd.DataFrame(
        [
            {"case_id": "c1", "classe_avocat": "A", "justification_avocat": "Règle A."},
            {"case_id": "c2", "classe_avocat": "E", "justification_avocat": "Aucun lien."},
        ]
    )
    key = pd.DataFrame(
        [
            {"case_id": "c1", "classe": "A", "sampling_weight": 1.0},
            {"case_id": "c2", "classe": "A", "sampling_weight": 1.0},
        ]
    )
    report = SUMMARY.summarize_agreement(annotations, key, expected_count=2, n_bootstrap=20)
    assert report["gain_agreement"] == 0.5
    assert report["positive_precision"] == 0.5
    assert report["gate_status"] == "failed"


def test_incomplete_annotations_do_not_produce_partial_metrics():
    annotations = pd.DataFrame(
        [{"case_id": "c1", "classe_avocat": "", "justification_avocat": ""}]
    )
    key = pd.DataFrame([{"case_id": "c1", "classe": "A", "sampling_weight": 1.0}])
    assert SUMMARY.summarize_agreement(annotations, key, expected_count=1)["status"] == "incomplete_annotations"
