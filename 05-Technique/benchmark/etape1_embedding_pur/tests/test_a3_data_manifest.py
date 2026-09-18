from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "108_build_a3_data_manifest.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("a3_data_manifest", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return _sha256(path)


def test_build_manifest_hashes_every_a3_campaign_data_input_once(tmp_path):
    manifest = _load_module()
    data_file = tmp_path / "data" / "inputs" / "items.bin"
    data_file.parent.mkdir(parents=True)
    data_file.write_bytes(b"sealed A3 input")
    digest = _sha256(data_file)
    a3_path = tmp_path / "a3.json"
    a3_digest = _write_json(a3_path, {"manifest_id": "a3", "datasets": {}})
    campaign_path = tmp_path / "campaign.json"
    campaign_digest = _write_json(
        campaign_path,
        {
            "campaign_id": "b1-a3",
            "a3": {"manifest_path": "a3.json", "sha256": a3_digest},
            "datasets": {
                "train": {"path": "inputs/items.bin", "sha256": digest},
                "evaluation": {"path": "inputs/items.bin", "sha256": digest},
            },
            "fold_inputs": {},
            "candidate_inputs": {},
            "graph_inputs": {},
        },
    )

    result = manifest.build_data_manifest(
        a3_path=a3_path,
        campaign_path=campaign_path,
        data_root=tmp_path / "data",
        repo_root=tmp_path,
    )

    assert result["campaign_manifest_sha256"] == campaign_digest
    assert result["a3_manifest_sha256"] == a3_digest
    assert result["campaign_manifest_path"] == "campaign.json"
    assert result["a3_manifest_path"] == "a3.json"
    assert result["artifact_count"] == 1
    assert result["artifacts"] == [
        {
            "path": "inputs/items.bin",
            "roles": ["datasets.evaluation", "datasets.train"],
            "size_bytes": len(b"sealed A3 input"),
            "sha256": digest,
            "expected_sha256": digest,
            "hash_matches": True,
            "redistribution": "not_cleared",
            "recovery": "Obtain or reconstruct from the authorized source, preserve the relative path, then run the A3 preflight.",
        }
    ]


def test_build_manifest_rejects_a_campaign_input_with_a_different_hash(tmp_path):
    manifest = _load_module()
    data_file = tmp_path / "data" / "items.bin"
    data_file.parent.mkdir()
    data_file.write_bytes(b"actual")
    a3_path = tmp_path / "a3.json"
    a3_digest = _write_json(a3_path, {"datasets": {}})
    campaign_path = tmp_path / "campaign.json"
    _write_json(
        campaign_path,
        {
            "campaign_id": "b1-a3",
            "a3": {"manifest_path": "a3.json", "sha256": a3_digest},
            "datasets": {"train": {"path": "items.bin", "sha256": "0" * 64}},
            "fold_inputs": {},
            "candidate_inputs": {},
            "graph_inputs": {},
        },
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        manifest.build_data_manifest(
            a3_path=a3_path,
            campaign_path=campaign_path,
            data_root=tmp_path / "data",
            repo_root=tmp_path,
        )
