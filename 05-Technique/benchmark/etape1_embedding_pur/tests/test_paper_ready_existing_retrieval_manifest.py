import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "paper_ready_existing_retrieval_a3_r3.json"
SUCCESSOR_MANIFEST = ROOT / "configs" / "paper_ready_existing_retrieval_a3_r4.json"
HISTORICAL_DEPTH_CURVES_SHA256 = "33219ad466a07c8be9800652913406b6c32399b84fb7457bd94555bafc53b10d"


def test_paper_ready_existing_retrieval_manifest_keeps_a3_and_b1_r1_provenance_frozen():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["campaign_id"] == "b1-paper-ready-existing-retrieval-a3-r3-2026-09-07"
    assert payload["supersedes"]["manifest_sha256"] == "bd303c1bc9825ad16743c54794d0899d63cbe19d4c7b91d4875fd0463dbf6175"
    assert payload["metrics"]["depth_curve_dispersion"] == "standard_deviation_across_seed_means_after_per_question_aggregation"
    assert payload["a3"]["sha256"] == "c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3"
    assert payload["source_campaign"]["manifest_sha256"] == "1b612a182742244dad59006e6d01b826a0285f01123aeeae67321b48c9de5e9a"
    assert payload["source_campaign"]["immutable"] is True
    assert payload["datasets"]["evaluation"]["questions"] == 754
    assert payload["candidate_universe"]["articles"]["count"] == 13236
    assert payload["candidate_universe"]["jurisprudence"]["count"] == 114851


def test_paper_ready_existing_retrieval_manifest_hashes_only_the_frozen_a3_rankings():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rankings = payload["frozen_rankings"]

    assert rankings["cosine"]["sha256"] == "7de0504d5e7a7a7caef303d0beb8019c28faac7dec1f38997e581ca782597334"
    assert rankings["ppr"]["sha256"] == "8f15c4d3e1ba465af7030e24f0e78772cae776fe8f12f5020f578d600964a938"
    assert {entry["top_k"] for entry in rankings.values()} == {100}
    assert payload["outputs"]["root"] != payload["source_campaign"]["manifest_path"]
    assert "paper_ready_existing_retrieval" in payload["outputs"]["root"]


def test_r3_manifest_preserves_the_historical_derivation_script_hash():
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["code_bundle"]["depth_curves"]["sha256"] == HISTORICAL_DEPTH_CURVES_SHA256


def test_r4_manifest_pins_the_current_derivation_script_and_supersedes_r3():
    payload = json.loads(SUCCESSOR_MANIFEST.read_text(encoding="utf-8"))
    script = ROOT / "scripts" / "97_build_b1_depth_curves.py"

    assert payload["code_bundle"]["depth_curves"]["path"] == str(script.relative_to(ROOT.parents[2]))
    assert payload["code_bundle"]["depth_curves"]["sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert payload["supersedes"]["manifest_path"] == str(MANIFEST.relative_to(ROOT.parents[2]))
    assert payload["supersedes"]["manifest_sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
