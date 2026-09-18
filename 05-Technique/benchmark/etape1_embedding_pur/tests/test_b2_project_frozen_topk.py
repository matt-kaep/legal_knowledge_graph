import importlib.util
from pathlib import Path
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "115_project_frozen_topk.py"


def _load_projector():
    spec = importlib.util.spec_from_file_location("b2_project_frozen_topk", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ProjectFrozenTopKTest(unittest.TestCase):
    def test_keeps_only_top_k_with_each_modality_coverage_intact(self):
        """A future projection that loses a modality or its original ordering must fail."""
        projector = _load_projector()
        ranking = pd.DataFrame([
            {"qid": "q1", "modality": "art", "rank": 1, "item_id": "a1"},
            {"qid": "q1", "modality": "art", "rank": 2, "item_id": "a2"},
            {"qid": "q1", "modality": "art", "rank": 3, "item_id": "a3"},
            {"qid": "q1", "modality": "jp", "rank": 1, "item_id": "j1"},
            {"qid": "q1", "modality": "jp", "rank": 2, "item_id": "j2"},
            {"qid": "q1", "modality": "jp", "rank": 3, "item_id": "j3"},
        ])

        projected = projector.project_top_k(ranking, k=2, condition_columns=["modality"])

        self.assertEqual(projected[["modality", "rank", "item_id"]].to_dict("records"), [
            {"modality": "art", "rank": 1, "item_id": "a1"},
            {"modality": "art", "rank": 2, "item_id": "a2"},
            {"modality": "jp", "rank": 1, "item_id": "j1"},
            {"modality": "jp", "rank": 2, "item_id": "j2"},
        ])

    def test_rejects_duplicate_candidate_inside_one_top_k_list(self):
        """A future projection that admits duplicate candidates must fail."""
        projector = _load_projector()
        ranking = pd.DataFrame([
            {"qid": "q1", "modality": "art", "rank": 1, "item_id": "a1"},
            {"qid": "q1", "modality": "art", "rank": 2, "item_id": "a1"},
        ])

        with self.assertRaisesRegex(ValueError, "duplicate candidates"):
            projector.project_top_k(ranking, k=2, condition_columns=["modality"])


if __name__ == "__main__":
    unittest.main()
