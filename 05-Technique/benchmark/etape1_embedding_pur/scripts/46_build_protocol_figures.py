from __future__ import annotations

import pandas as pd


def prepare_lightgcn_history(df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["epoch", "graph_version", "variant"]
    series_order = [
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
    ]
    value_cols = [column for column in series_order if column in df.columns]
    out = df.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="series",
        value_name="value",
    )
    return out[["epoch", "series", "value", "graph_version", "variant"]]
