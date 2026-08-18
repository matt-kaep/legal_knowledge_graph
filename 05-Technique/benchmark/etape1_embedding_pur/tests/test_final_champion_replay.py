import importlib.util
import hashlib
import json
from pathlib import Path

import pytest
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "45_run_final_champions.py"
spec = importlib.util.spec_from_file_location("final_grouped_replay", SCRIPT)
final_replay = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(final_replay)


def _trained_champion():
    return {
        "method": "LightGCN-trained_K2-s42-lr0.001-e30-la1-neg-random",
        "modality": "art",
        "variant": "trained_K2",
        "train_k": 2,
        "seed": 42,
        "lr": 0.001,
        "epochs": 30,
        "selected_epoch_index": 6,
        "replay_epochs": 7,
        "lambda_anchor": 1.0,
        "negative_sampling_strategy": "random",
        "eligible_champion": True,
        "n_folds_covered": 5,
        "question_coverage": 1.0,
        "protocol_version": "grouped_v2",
        "dataset_sha256": "dataset-hash",
        "fold_assignment_sha256": "fold-hash",
        "robustness_summary_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "robustness_summary_path": "/dev/null",
    }


def test_grouped_replay_rejects_champion_without_fixed_epoch():
    champion = _trained_champion()
    champion.pop("replay_epochs")

    with pytest.raises(ValueError, match="replay_epochs"):
        final_replay.validate_grouped_champion_bundle(
            {"lightgcn": {"art": champion}},
            dataset_sha256="dataset-hash",
            fold_assignment_sha256="fold-hash",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("eligible_champion", False),
        ("n_folds_covered", 4),
        ("question_coverage", 0.99),
        ("dataset_sha256", "wrong"),
        ("fold_assignment_sha256", "wrong"),
    ],
)
def test_grouped_replay_rejects_incomplete_or_mismatched_champion(field, value):
    champion = _trained_champion()
    champion[field] = value

    with pytest.raises(ValueError, match=field):
        final_replay.validate_grouped_champion_bundle(
            {"lightgcn": {"art": champion}},
            dataset_sha256="dataset-hash",
            fold_assignment_sha256="fold-hash",
        )


def test_lightgcn_replay_args_force_fixed_final_epoch():
    args = final_replay.build_lightgcn_replay_args(
        _trained_champion(),
        train_bench_dir=Path("/train"),
        eval_bench_dir=Path("/eval"),
        graph_version="G1",
        suffix="article",
        top_k_out=1000,
    )

    assert args[args.index("--epochs") + 1] == "7"
    assert args[args.index("--checkpoint-selection") + 1] == "fixed_final_epoch"
    assert args[args.index("--top-k-out") + 1] == "1000"


def test_grouped_replay_roots_never_overlap_legacy_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(final_replay.graph_protocol, "BENCH_ROOT", tmp_path)

    cv_root, out_dir = final_replay.resolve_replay_roots("G1", "grouped_v2")

    assert cv_root == tmp_path / "_cv_grouped_v2" / "G1"
    assert out_dir == tmp_path / "_final_grouped_v2" / "G1"
    assert "_cv/" not in f"{cv_root}/"
    assert "_final_champions/" not in f"{out_dir}/"


@pytest.mark.parametrize(
    ("graph_version", "expected"),
    [("G1", "g1"), ("G6-citation-AA-knn5", "g1"), ("G7-citation-JJ-cit1-sem025-knn5", "g1"), ("G3", "g3")],
)
def test_derived_graph_coverage_uses_g1_source_row(graph_version, expected):
    assert final_replay.coverage_source_graph_key(graph_version) == expected


def test_target_specific_replay_epochs_are_not_deduplicated():
    article = _trained_champion()
    article["selection_target"] = "art"
    jp = {**article, "modality": "jp", "selection_target": "jp", "replay_epochs": 11}

    unique = final_replay.unique_lightgcn_champions({"art": article, "jp": jp})

    assert len(unique) == 2
    assert {row["replay_epochs"] for row in unique} == {7, 11}


def test_direct_grouped_replay_revalidates_campaign_inputs():
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "confirmatory_campaign_grouped_v2_repro_v1.json"
    campaign = json.loads(manifest_path.read_text())
    campaign["datasets"]["internal_eval"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="sha256 mismatch"):
        final_replay.validate_campaign_provenance(campaign)


def test_direct_grouped_replay_honors_resource_gate():
    manifest_path = Path(__file__).resolve().parents[1] / "configs" / "confirmatory_campaign_grouped_v2_repro_v1.json"
    campaign = json.loads(manifest_path.read_text())
    campaign["code_bundle"]["final_replay"]["sha256"] = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    campaign["resources"]["ram_minimum_gb_per_graph_job"] = None

    with pytest.raises(RuntimeError, match="ram_minimum_unmeasured"):
        final_replay.validate_campaign_provenance(campaign)


def test_champion_mask_matches_cv_budget_and_replay_count_separately():
    champion = {**_trained_champion(), "selection_target": "art"}
    run_id = final_replay.lightgcn_run_id(champion)
    rows = pd.DataFrame([
        {"method": "LightGCN-trained_K2", "modality": "art", "variant": "trained_K2", "train_k": 2, "seed": 42, "lr": 0.001, "cv_epochs": 30, "replay_epochs": 7, "lambda_anchor": 1.0, "selection_target": "art", "negative_sampling_strategy": "random", "run_id": run_id},
        {"method": "LightGCN-trained_K2", "modality": "art", "variant": "trained_K2", "train_k": 2, "seed": 42, "lr": 0.001, "cv_epochs": 30, "replay_epochs": 11, "lambda_anchor": 1.0, "selection_target": "jp", "negative_sampling_strategy": "random", "run_id": "other"},
    ])

    selected = final_replay._select_champion_rows(rows, {"art": champion})

    assert len(selected) == 1
    assert selected.iloc[0]["replay_epochs"] == 7
