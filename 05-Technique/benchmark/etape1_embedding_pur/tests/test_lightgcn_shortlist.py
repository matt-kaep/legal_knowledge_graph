import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "65_select_lightgcn_shortlist.py"
spec = importlib.util.spec_from_file_location("lightgcn_shortlist", SCRIPT)
shortlist = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(shortlist)
CONTROLS = ["G1", "G6-citation-AA-knn5", "G6-citation-JJ-knn5"]


def _rows():
    rows = []
    for graph_id, article_recall, jp_hit in [
        ("G1", 0.50, 0.20),
        ("G6-citation-AA-knn5", 0.55, 0.22),
        ("G6-citation-JJ-knn5", 0.54, 0.23),
        ("G7-A", 0.60, 0.21),
        ("G7-B", 0.58, 0.27),
    ]:
        rows.extend(
            [
                {"graph_id": graph_id, "modality": "art", "eligible_champion": True, "article_recall_at_10_mean": article_recall, "article_ndcg_at_10_mean": 0.4, "article_mrr_at_10_mean": 0.3},
                {"graph_id": graph_id, "modality": "jp", "eligible_champion": True, "jp_hit_at_10_mean": jp_hit, "jp_ndcg_at_10_mean": 0.3, "jp_mrr_at_10_mean": 0.2},
            ]
        )
    return pd.DataFrame(rows)


def test_shortlist_keeps_controls_and_distinct_best_g7_targets():
    selected = shortlist.select_shortlist(_rows(), always_include=CONTROLS)

    assert selected == [
        "G1",
        "G6-citation-AA-knn5",
        "G6-citation-JJ-knn5",
        "G7-A",
        "G7-B",
    ]


def test_shortlist_deduplicates_same_g7_winner():
    rows = _rows()
    rows.loc[rows["graph_id"] == "G7-A", "jp_hit_at_10_mean"] = 0.30

    selected = shortlist.select_shortlist(rows, always_include=CONTROLS)

    assert selected == ["G1", "G6-citation-AA-knn5", "G6-citation-JJ-knn5", "G7-A"]


def test_shortlist_ignores_ineligible_high_score():
    rows = _rows()
    rows.loc[rows["graph_id"] == "G7-A", "eligible_champion"] = False

    selected = shortlist.select_shortlist(rows, always_include=CONTROLS)

    assert "G7-A" not in selected
    assert "G7-B" in selected


def test_shortlist_loader_rejects_manifest_mismatch(tmp_path):
    path = tmp_path / "shortlist.json"
    path.write_text('{"manifest_sha256":"wrong","graph_ids":["G1"]}')

    try:
        shortlist.load_frozen_shortlist(path, "expected")
    except ValueError as exc:
        assert "manifest_sha256" in str(exc)
    else:
        raise AssertionError("manifest mismatch should be rejected")


def test_screening_summary_rejects_wrong_fold_hash():
    manifest = {
        "protocol_version": "grouped_v2",
        "datasets": {"train": {"sha256": "dataset"}},
        "folds": {"sha256": "folds", "count": 5},
    }
    frame = pd.DataFrame([
        {"modality": modality, "eligible_champion": True, "protocol_version": "grouped_v2", "dataset_sha256": "dataset", "fold_assignment_sha256": "wrong", "n_folds_covered": 5, "question_coverage": 1.0}
        for modality in ("art", "jp")
    ])

    with pytest.raises(ValueError, match="fold_assignment_sha256"):
        shortlist.validate_screening_summary(frame, manifest, "G7")
