from pathlib import Path
import importlib.util

import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "46_build_protocol_figures.py"
)
spec = importlib.util.spec_from_file_location("protocol_figures", SCRIPT)
protocol_figures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol_figures)


def test_lightgcn_history_long_format_is_plot_ready():
    df = pd.DataFrame(
        [
            {
                "epoch": 0,
                "train_loss": 1.2,
                "val_hit": 0.31,
                "graph_version": "G0",
                "variant": "trained_K2",
            },
            {
                "epoch": 1,
                "train_loss": 1.0,
                "val_hit": 0.34,
                "graph_version": "G0",
                "variant": "trained_K2",
            },
        ]
    )

    out = protocol_figures.prepare_lightgcn_history(df)

    assert list(out.columns) == [
        "epoch",
        "series",
        "value",
        "graph_version",
        "variant",
    ]
    assert set(out["series"]) == {"train_loss", "val_hit"}


def test_lightgcn_history_supports_real_exported_columns():
    df = pd.DataFrame(
        [
            {
                "epoch": 0,
                "graph_version": "canonical",
                "variant": "trained_K2",
                "train_loss": 1.2,
                "bpr_loss": 1.0,
                "anchor_loss": 0.2,
                "val_hit": 0.31,
                "val_ndcg": 0.22,
                "val_mrr": 0.19,
                "val_recall": 0.31,
                "val_norm_rank": 0.22,
                "val_hit_jp": 0.41,
                "val_ndcg_jp": 0.29,
                "fold": 0,
                "seed": 42,
            },
        ]
    )

    out = protocol_figures.prepare_lightgcn_history(df)

    assert set(out["series"]) == {
        "train_loss",
        "bpr_loss",
        "anchor_loss",
        "val_hit",
        "val_ndcg",
        "val_mrr",
        "val_recall",
        "val_norm_rank",
        "val_hit_jp",
        "val_ndcg_jp",
    }
    assert out["value"].notna().all()
