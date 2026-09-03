import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(script_name: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / script_name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lightgcn = _load("32_lightgcn_strict.py", "lightgcn_checkpoint_selection")
cv_lightgcn = _load("44_run_cv_lightgcn.py", "cv_lightgcn_checkpoint_selection")


def test_validation_best_calls_epoch_evaluator():
    calls = []

    def evaluate_fn(item_final, variant, *, top_k_out):
        calls.append((item_final, variant, top_k_out))
        return [{"hit_strict_art": 0.4}], []

    rows = lightgcn.evaluate_training_epoch(
        "validation_best",
        item_final="embedding",
        variant="trained_K2",
        top_k_out=10,
        evaluate_fn=evaluate_fn,
    )

    assert rows == [{"hit_strict_art": 0.4}]
    assert calls == [("embedding", "trained_K2", 10)]


def test_fixed_final_epoch_does_not_call_epoch_evaluator():
    def forbidden_evaluate(*args, **kwargs):
        raise AssertionError("fixed_final_epoch must not inspect eval during training")

    rows = lightgcn.evaluate_training_epoch(
        "fixed_final_epoch",
        item_final="embedding",
        variant="trained_K2",
        top_k_out=10,
        evaluate_fn=forbidden_evaluate,
    )

    assert rows is None


def test_replay_epoch_is_selected_from_fold_means_not_history_row_counts():
    history = pd.DataFrame(
        [
            {"fold": 0, "epoch": 1, "val_recall": 0.9},
            {"fold": 0, "epoch": 1, "val_recall": 0.9},
            {"fold": 0, "epoch": 2, "val_recall": 0.8},
            {"fold": 1, "epoch": 1, "val_recall": 0.0},
            {"fold": 1, "epoch": 2, "val_recall": 0.4},
        ]
    )

    assert cv_lightgcn.select_replay_epoch(history, "val_recall") == 3


def test_replay_epoch_tie_prefers_earlier_epoch():
    history = pd.DataFrame(
        [
            {"fold": fold, "epoch": epoch, "val_hit_jp": 0.5}
            for fold in range(5)
            for epoch in (3, 4)
        ]
    )

    assert cv_lightgcn.select_replay_epoch(history, "val_hit_jp") == 4


def test_selection_metric_is_target_specific():
    assert cv_lightgcn.selection_metric_for_target("art") == "val_recall"
    assert cv_lightgcn.selection_metric_for_target("jp") == "val_hit_jp"


def test_resolve_device_rejects_requested_cuda_without_cuda(monkeypatch):
    monkeypatch.setattr(lightgcn.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        lightgcn.resolve_device("cuda")


def test_attach_replay_epochs_uses_only_matching_champion_history():
    champions = {
        "art": {
            "variant": "trained_K2",
            "train_k": 2,
            "seed": 42,
            "lr": 0.001,
            "epochs": 3,
            "lambda_anchor": 1.0,
            "negative_sampling_strategy": "random",
            "graph_version": "G1",
            "selection_target": "art",
        },
        "jp": {
            "variant": "trained_K2",
            "train_k": 2,
            "seed": 42,
            "lr": 0.001,
            "epochs": 3,
            "lambda_anchor": 1.0,
            "negative_sampling_strategy": "random",
            "graph_version": "G1",
            "selection_target": "jp",
        },
    }
    history = pd.DataFrame(
        [
            {
                "fold": fold,
                "epoch": epoch,
                "variant": "trained_K2",
                "train_k": 2,
                "seed": 42,
                "lr": 0.001,
                "epochs": 3,
                "lambda_anchor": 1.0,
                "negative_sampling_strategy": "random",
                "graph_version": "G1",
                "selection_target": target,
                "val_recall": 0.8 if target == "art" and epoch == 1 else 0.2,
                "val_hit_jp": 0.9 if target == "jp" and epoch == 2 else 0.1,
            }
            for target in ("art", "jp")
            for fold in range(5)
            for epoch in (1, 2)
        ]
    )

    enriched = cv_lightgcn.attach_replay_epochs(champions, history)

    assert enriched["art"]["selected_epoch_index"] == 1
    assert enriched["art"]["replay_epochs"] == 2
    assert enriched["art"]["epoch_selection_metric"] == "val_recall"
    assert enriched["jp"]["selected_epoch_index"] == 2
    assert enriched["jp"]["replay_epochs"] == 3
    assert enriched["jp"]["epoch_selection_metric"] == "val_hit_jp"
