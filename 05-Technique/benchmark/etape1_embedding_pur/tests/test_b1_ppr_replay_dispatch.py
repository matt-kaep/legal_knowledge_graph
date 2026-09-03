"""Regression checks for the append-only B1-r1 PPR replay dispatch."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "b1_a3_r1_ppr_replay_dispatch.json"
WRAPPER = ROOT / "scripts" / "sbatch_b1_a3_r1_ppr_replay.sh"
PARENT_MANIFEST = ROOT / "configs" / "confirmatory_campaign_b1_a3_r1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_ppr_replay_dispatch_is_bound_to_the_frozen_b1_r1_selection():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["parent_campaign"]["path"] == str(PARENT_MANIFEST.relative_to(ROOT.parents[2]))
    assert payload["parent_campaign"]["sha256"] == _sha256(PARENT_MANIFEST)
    assert payload["frozen_champions"] == {
        "path": (
            "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/"
            "_campaign_b1_a3_effective_retrieval_r1_20260902/frozen/ppr_champions.json"
        ),
        "sha256": "61e80ca1d1f746f4c759c6c118bb79fa605ed6c869bb04498b9b6abeee48db6d",
        "selection_data": "train_cv_only",
    }
    assert payload["evaluation"] == {
        "split": "eval_rich_retrievable_strict",
        "questions": 754,
        "sha256": "850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59",
    }
    assert payload["top_k_out"] == 100
    assert {"E017", "E021", "E022"}.issubset(payload["historical_experiments_excluded"])

    for item in payload["code_bundle"].values():
        path = ROOT.parents[2] / item["path"]
        assert _sha256(path) == item["sha256"]


def test_ppr_replay_wrapper_runs_only_the_frozen_parent_campaign():
    script = WRAPPER.read_text(encoding="utf-8")

    assert "#SBATCH --partition=CPU" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=12G" in script
    assert "#SBATCH --time=08:00:00" in script
    assert "confirmatory_campaign_b1_a3_r1.json" in script
    assert "--family ppr" in script
    assert "96_replay_b1_a3_champions.py" in script
