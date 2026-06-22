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
