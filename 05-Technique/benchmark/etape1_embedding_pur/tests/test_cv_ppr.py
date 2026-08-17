import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "43_run_cv_ppr.py"
spec = importlib.util.spec_from_file_location("cv_ppr_fold_first", SCRIPT)
cv_ppr = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cv_ppr)


def test_ppr_summary_exposes_fold_first_primary_metrics():
    raw = pd.DataFrame(
        [
            {
                "fold": fold,
                "qid": f"q{fold}",
                "k_in": 10,
                "seed_variant": "both",
                "alpha": 0.5,
                "hit_strict_art": 0.5 + fold / 100,
                "ndcg_strict_art": 0.4,
                "mrr_strict_art": 0.3,
                "m1_strict_art": 0.6,
            }
            for fold in range(5)
        ]
    )

    fold_metrics, summary = cv_ppr.summarize_cv_outputs(raw, "art")

    assert len(fold_metrics) == 5
    assert summary.loc[0, "article_hit_at_10_mean"] == 0.52
    assert summary.loc[0, "article_ndcg_at_10_mean"] == 0.4
    assert summary.loc[0, "article_mrr_at_10_mean"] == 0.3
    assert summary.loc[0, "article_recall_at_10_mean"] == 0.6
    assert summary.loc[0, "eligible_champion"]


def test_ppr_uses_grouped_v2_fold_namespace(monkeypatch):
    captured = {}

    def fake_load_verified_grouped_fold_assignments(
        bench_dir,
        split=cv_ppr.graph_protocol.OFFICIAL_TRAIN_SPLIT,
        version=cv_ppr.graph_protocol.PROTOCOL_VERSION,
    ):
        captured["bench_dir"] = bench_dir
        captured["split"] = split
        captured["version"] = version
        return (
            pd.DataFrame({"qid": [f"q{fold}" for fold in range(5)], "fold": range(5)}),
            {"protocol_version": version},
        )

    monkeypatch.setattr(
        cv_ppr.graph_protocol,
        "load_verified_grouped_fold_assignments",
        fake_load_verified_grouped_fold_assignments,
    )

    cv_ppr.load_fold_assignments(Path("/tmp/bench"), {f"q{fold}" for fold in range(5)})

    assert captured["bench_dir"] == Path("/tmp/bench")
    assert captured["version"] == cv_ppr.graph_protocol.PROTOCOL_VERSION


@pytest.mark.parametrize("mismatch_key", ["dataset_sha256", "fold_assignment_sha256"])
def test_ppr_rejects_grouped_v2_fold_metadata_hash_mismatch(tmp_path, monkeypatch, mismatch_key):
    bench_root = tmp_path / "bench"
    bench_dir = bench_root / "G0" / "train_augmented_retrievable_strict"
    bench_dir.mkdir(parents=True)
    bench_global = bench_dir / "bench_global.json"
    bench_global.write_text(json.dumps({"questions": [{"qid": f"q{fold}"} for fold in range(5)]}))
    protocol_dir = bench_root / "_protocol" / "grouped_v2" / "train_augmented_retrievable_strict"
    protocol_dir.mkdir(parents=True)
    assignments_path = protocol_dir / "fold_assignments.csv"
    pd.DataFrame({"qid": [f"q{fold}" for fold in range(5)], "fold": range(5)}).to_csv(assignments_path, index=False)
    metadata = {
        "protocol_version": "grouped_v2",
        "dataset_sha256": hashlib.sha256(bench_global.read_bytes()).hexdigest(),
        "fold_assignment_sha256": hashlib.sha256(assignments_path.read_bytes()).hexdigest(),
    }
    metadata[mismatch_key] = "not-the-real-hash"
    (protocol_dir / "fold_metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(cv_ppr.graph_protocol, "BENCH_ROOT", bench_root)

    with pytest.raises(ValueError, match=mismatch_key):
        cv_ppr.load_fold_assignments(bench_dir, {f"q{fold}" for fold in range(5)})


def test_ppr_materializes_paired_deltas_by_config_and_fold():
    candidate = pd.DataFrame(
        [
            {"fold": fold, "k_in": 10, "seed_variant": "both", "alpha": 0.5, "hit_strict_art": 0.6 + fold / 100}
            for fold in [4, 2, 0, 3, 1]
        ]
    )
    control = pd.DataFrame(
        [
            {"fold": fold, "k_in": 10, "seed_variant": "both", "alpha": 0.5, "hit_strict_art": 0.5 + fold / 100}
            for fold in [1, 3, 0, 4, 2]
        ]
    )

    deltas = cv_ppr.build_paired_deltas(candidate, control, "art")

    assert deltas.loc[0, "eligible_comparison"]
    assert deltas.loc[0, "hit_strict_art_delta_mean"] == pytest.approx(0.1)

    missing = cv_ppr.build_paired_deltas(candidate, control[control["fold"] != 4], "art")

    assert not missing.loc[0, "eligible_comparison"]
