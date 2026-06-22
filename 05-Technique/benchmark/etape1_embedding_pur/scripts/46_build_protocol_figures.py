from __future__ import annotations

import pandas as pd


def prepare_lightgcn_history(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["epoch", "graph_version", "variant"]
    value_cols = [c for c in ["train_loss", "val_hit", "val_ndcg"] if c in df.columns]
    out = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="series",
        value_name="value",
    )
    return out[["epoch", "series", "value", "graph_version", "variant"]]
