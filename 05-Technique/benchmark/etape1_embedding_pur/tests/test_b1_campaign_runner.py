import importlib.util
import hashlib
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "94_run_b1_a3_campaign.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("b1_campaign_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _payload():
    return {
        "campaign_id": "b1-a3",
        "protocol_version": "grouped_v5_no_eval_overlap_effective_retrieval_v3",
        "datasets": {
            "train": {"split": "train-a3"},
            "evaluation": {"split": "eval-a3"},
        },
        "folds": {"count": 5},
        "graphs": ["G1", "G6"],
        "parameters": {
            "cosine": {"top_k_out": 100},
            "ppr": {"k_in": [5], "seed_variant": ["both"], "alpha": [0.5]},
            "lightgcn": {
                "cv_seeds": [42],
                "train_k": [2],
                "learning_rate": [0.001],
                "lambda_anchor": [1.0],
                "epochs": 30,
                "negative_sampling_strategy": ["random"],
            },
        },
        "outputs": {
            "root": "data/b1-a3",
            "cosine": "data/b1-a3/cosine",
            "ppr_cv": "data/b1-a3/ppr-cv",
            "lightgcn_cv": "data/b1-a3/lightgcn-cv",
        },
    }


def test_b1_commands_bind_a3_inputs_and_fresh_output_namespaces(tmp_path):
    runner = _load_runner()
    payload = _payload()

    cosine = runner.build_stage_commands(payload, "cosine", data_root=tmp_path)[0]
    ppr = runner.build_stage_commands(payload, "ppr-cv", graph_id="G6", data_root=tmp_path)[0]
    lightgcn = runner.build_stage_commands(payload, "lightgcn-cv", graph_id="G6", data_root=tmp_path)[0]

    assert "--top-k-out" in cosine
    assert cosine[cosine.index("--top-k-out") + 1] == "100"
    assert str(tmp_path / "data/b1-a3/cosine") in cosine

    assert ppr[ppr.index("--split") + 1] == "train-a3"
    assert ppr[ppr.index("--protocol-version") + 1] == "grouped_v5_no_eval_overlap_effective_retrieval_v3"
    assert str(tmp_path / "data/b1-a3/ppr-cv/G6") in ppr

    assert lightgcn[lightgcn.index("--split") + 1] == "train-a3"
    assert lightgcn[lightgcn.index("--seed") + 1] == "42"
    assert str(tmp_path / "data/b1-a3/lightgcn-cv/G6") in lightgcn


def test_manifest_file_hash_is_the_external_campaign_reference(tmp_path):
    runner = _load_runner()
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({"campaign_id": "b1-a3", "z": 1}, indent=2), encoding="utf-8")

    assert runner._sha256(manifest) == hashlib.sha256(manifest.read_bytes()).hexdigest()
