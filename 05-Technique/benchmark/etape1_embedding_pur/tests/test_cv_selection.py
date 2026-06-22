from pathlib import Path
import importlib.util

import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "42_run_cv_b3_b4.py"
)
spec = importlib.util.spec_from_file_location("cv_b3b4", SCRIPT)
cv_b3b4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_b3b4)


def test_select_champion_uses_hit_then_ndcg_then_mrr():
    df = pd.DataFrame(
        [
            {
                "method": "B3-e",
                "k_in": 10,
                "modality": "art",
                "hit_strict": 0.50,
                "ndcg_strict": 0.40,
                "mrr_strict": 0.35,
                "m1_strict": 0.51,
                "m2_strict": 0.42,
            },
            {
                "method": "B3-e",
                "k_in": 20,
                "modality": "art",
                "hit_strict": 0.50,
                "ndcg_strict": 0.41,
                "mrr_strict": 0.34,
                "m1_strict": 0.50,
                "m2_strict": 0.41,
            },
        ]
    )
    best = cv_b3b4.select_champion(df, "art")
    assert best["k_in"] == 20
