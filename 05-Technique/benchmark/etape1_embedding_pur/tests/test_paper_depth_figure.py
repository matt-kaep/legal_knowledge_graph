from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "116_build_paper_depth_figure.py"


def _curve_rows(source: str, target: str, value_at_ten: float, *, seeds: int) -> list[dict]:
    return [
        {
            "source": source,
            "target": target,
            "k": k,
            "mean": value_at_ten + (k - 10) * 0.001,
            "seed_std": 0.0,
            "seeds": seeds,
        }
        for k in range(1, 101)
    ]


def _reranking_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed: list[dict] = []
    means: list[dict] = []
    seeds = {"cosine": [None], "ppr": [None], "lightgcn": [42, 43, 44]}
    values = {
        ("cosine", "article"): 0.570,
        ("ppr", "article"): 0.666,
        ("lightgcn", "article"): 0.654,
        ("cosine", "jp"): 0.298,
        ("ppr", "jp"): 0.309,
        ("lightgcn", "jp"): 0.329,
    }
    for family, family_seeds in seeds.items():
        for modality in ("article", "jp"):
            for k_in in (10, 20, 30, 40, 50, 60, 70):
                value = values[(family, modality)] if k_in == 70 else values[(family, modality)] - 0.01
                for seed in family_seeds:
                    per_seed.append(
                        {
                            "family": family,
                            "modality": modality,
                            "k_in": k_in,
                            "replay_seed": seed,
                            "hit_at_10": value,
                        }
                    )
                means.append(
                    {
                        "family": family,
                        "modality": modality,
                        "k_in": k_in,
                        "hit_at_10": value,
                        "seed_count": len(family_seeds),
                    }
                )
    return pd.DataFrame(per_seed), pd.DataFrame(means)


class PaperDepthFigureTest(unittest.TestCase):
    def test_builds_four_panel_figure_and_preserves_ppr_g6_jp_identity(self):
        """Changing the JP E029 source label or omitting a frozen depth must fail this export."""
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            r5_rows = []
            for source, article_at_ten, jp_at_ten in (("cosine", 0.431510, 0.217949), ("ppr", 0.576881, 0.230659)):
                r5_rows.extend(_curve_rows(source, "articles", article_at_ten, seeds=1))
                r5_rows.extend(_curve_rows(source, "jurisprudence", jp_at_ten, seeds=1))
            g6_rows = _curve_rows("lightgcn_g6", "articles", 0.551069, seeds=3)
            g6_rows += _curve_rows("lightgcn_g6", "jurisprudence", 0.266357, seeds=3)
            ppr_g6_jp_rows = _curve_rows("ppr_g6_jurisprudence", "jurisprudence", 0.228227, seeds=1)
            per_seed, seed_mean = _reranking_rows()

            r5 = tmp_path / "retrieval_r5.csv"
            lightgcn = tmp_path / "retrieval_lightgcn.csv"
            ppr_g6_jp = tmp_path / "retrieval_ppr_g6_jp.csv"
            per_seed_path = tmp_path / "reranking_per_seed.csv"
            seed_mean_path = tmp_path / "reranking_seed_mean.csv"
            provenance_path = tmp_path / "provenance.json"
            pd.DataFrame(r5_rows).to_csv(r5, index=False)
            pd.DataFrame(g6_rows).to_csv(lightgcn, index=False)
            pd.DataFrame(ppr_g6_jp_rows).to_csv(ppr_g6_jp, index=False)
            per_seed.to_csv(per_seed_path, index=False)
            seed_mean.to_csv(seed_mean_path, index=False)
            provenance_path.write_text(json.dumps({"manifest_id": "fixture-provenance", "sources": {}}), encoding="utf-8")
            out_dir = tmp_path / "figure"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--retrieval-r5", str(r5),
                    "--retrieval-lightgcn", str(lightgcn),
                    "--retrieval-ppr-g6-jp", str(ppr_g6_jp),
                    "--reranking-per-seed", str(per_seed_path),
                    "--reranking-seed-mean", str(seed_mean_path),
                    "--provenance", str(provenance_path),
                    "--out-dir", str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            retrieval = pd.read_csv(out_dir / "retrieval_depth_tidy.csv")
            reranking = pd.read_csv(out_dir / "reranking_depth_tidy.csv")
            manifest = json.loads((out_dir / "depth_figure_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(retrieval.shape[0], 600)
            self.assertEqual(set(retrieval["k"]), set(range(1, 101)))
            self.assertEqual(
                retrieval.loc[
                    (retrieval["target"] == "judicial_decisions") & (retrieval["method"] == "ppr"), "graph_label"
                ].unique().tolist(),
                ["PPR G7-AA"],
            )
            self.assertEqual(reranking.shape[0], 42)
            self.assertEqual(set(reranking["k_in"]), {10, 20, 30, 40, 50, 60, 70})
            self.assertEqual(
                reranking.loc[
                    (reranking["target"] == "judicial_decisions") & (reranking["method"] == "ppr"), "graph_label"
                ].unique().tolist(),
                ["PPR G6-AA"],
            )
            self.assertEqual(
                reranking.loc[
                    (reranking["target"] == "judicial_decisions")
                    & (reranking["method"] == "ppr")
                    & (reranking["k_in"] == 70), "nhit_at_10"
                ].item(),
                0.309,
            )
            self.assertEqual(manifest["coverage"]["questions"], 754)
            self.assertEqual(manifest["provenance"]["manifest_id"], "fixture-provenance")
            self.assertEqual(manifest["canonical_source_contract"]["manifest_id"], "fixture-provenance")
            self.assertEqual(manifest["figure_layout"]["panels"], ["A", "B", "C", "D"])
            self.assertIn("PPR G6-AA", (out_dir / "README.md").read_text(encoding="utf-8"))
            self.assertTrue((out_dir / "paper_depth_figure.pdf").is_file())
            self.assertTrue((out_dir / "paper_depth_figure.png").is_file())
            with Image.open(out_dir / "paper_depth_figure.png") as png:
                self.assertEqual(png.size, (2400, 1500))
