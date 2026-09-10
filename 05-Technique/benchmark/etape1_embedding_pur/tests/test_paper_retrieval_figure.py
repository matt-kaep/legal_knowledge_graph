from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "118_build_paper_retrieval_figure.py"


class PaperRetrievalFigureTests(unittest.TestCase):
    def test_builder_emits_two_panel_retrieval_figure_with_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source = temp / "retrieval.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["target", "method", "graph_label", "k", "nhit_at_k", "seed_count"])
                writer.writeheader()
                for target in ("statutory_articles", "judicial_decisions"):
                    for method, label, scale in (
                        ("cosine", "BGE-M3 cosine", 0.62),
                        ("ppr", "PPR G6-AA" if target == "statutory_articles" else "PPR G7-AA", 0.74),
                        ("lightgcn", "LightGCN G6", 0.88),
                    ):
                        for k in range(1, 101):
                            writer.writerow({"target": target, "method": method, "graph_label": label, "k": k, "nhit_at_k": scale * k / 100, "seed_count": 3 if method == "lightgcn" else 1})

            out_dir = temp / "out"
            subprocess.run(
                [sys.executable, str(BUILDER), "--retrieval-csv", str(source), "--out-dir", str(out_dir)],
                check=True,
            )
            receipt = json.loads((out_dir / "retrieval_figure_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["layout"]["panel_count"], 2)
            self.assertEqual(receipt["coverage"]["rows"], 600)
            self.assertGreater(receipt["y_limits"]["judicial_decisions"][1], 0.88)
            self.assertTrue((out_dir / "paper_retrieval_figure.pdf").is_file())
            self.assertTrue((out_dir / "paper_retrieval_figure.png").is_file())


if __name__ == "__main__":
    unittest.main()
