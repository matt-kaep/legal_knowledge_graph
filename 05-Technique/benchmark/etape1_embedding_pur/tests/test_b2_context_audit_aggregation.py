import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "105_aggregate_b2_fulltext_context_audits.py"


def _load_aggregator():
    spec = importlib.util.spec_from_file_location("b2_context_audit_aggregation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _report(*, family: str, modality: str, k_in: int, compatible: bool = True, replay_seed: str | None = None) -> dict[str, object]:
    return {
        "schema_version": "b2-e029-fulltext-context-audit.v1",
        "model": {"id": "model", "revision": "revision", "tokenizer_files": {"tokenizer.json": "abc"}},
        "context_limit_tokens": 16384,
        "max_output_tokens": 256,
        "candidate_text_policy": "full_text_unmodified",
        "prompt_files": {"article": {"sha256": "article"}, "jp": {"sha256": "jp"}},
        "job_files": [{"path": f"/{family}-{modality}-{k_in}.jsonl", "jobs": 754, "sha256": f"jobs-{family}-{modality}-{k_in}"}],
        "conditions": [{
            "family": family,
            "modality": modality,
            "k_in": k_in,
            "questions": 754,
            "compatible": compatible,
            "overflow_qids": [] if compatible else ["q1"],
            **({"replay_seed": replay_seed} if replay_seed is not None else {}),
        }],
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_aggregate_preserves_hashed_shards_and_partitions_compatibility(tmp_path):
    aggregator = _load_aggregator()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _report(family="cosine", modality="article", k_in=10))
    _write(second, _report(family="ppr_g6", modality="jp", k_in=20, compatible=False))

    result = aggregator.aggregate_reports([first, second], expected_conditions=2)

    assert [item["sha256"] for item in result["source_reports"]] == [aggregator.sha256(first), aggregator.sha256(second)]
    assert [(row["family"], row["modality"], row["k_in"]) for row in result["conditions"]] == [
        ("cosine", "article", 10), ("ppr_g6", "jp", 20),
    ]
    assert result["compatible_conditions"] == 1
    assert result["incompatible_conditions"] == 1


def test_aggregate_rejects_duplicate_condition_across_receipts(tmp_path):
    aggregator = _load_aggregator()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _report(family="cosine", modality="article", k_in=10))
    _write(second, _report(family="cosine", modality="article", k_in=10))

    with pytest.raises(ValueError, match="duplicate condition"):
        aggregator.aggregate_reports([first, second], expected_conditions=2)


def test_aggregate_keeps_lightgcn_seeds_as_distinct_conditions(tmp_path):
    aggregator = _load_aggregator()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _report(family="lightgcn_g6", modality="jp", k_in=10, replay_seed="42"))
    _write(second, _report(family="lightgcn_g6", modality="jp", k_in=10, replay_seed="43"))

    result = aggregator.aggregate_reports([first, second], expected_conditions=2)

    assert [row["replay_seed"] for row in result["conditions"]] == ["42", "43"]


def test_aggregate_rejects_contract_mismatch_and_incomplete_matrix(tmp_path):
    aggregator = _load_aggregator()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(first, _report(family="cosine", modality="article", k_in=10))
    incompatible_contract = _report(family="ppr_g6", modality="jp", k_in=10)
    incompatible_contract["max_output_tokens"] = 128
    _write(second, incompatible_contract)

    with pytest.raises(ValueError, match="max_output_tokens"):
        aggregator.aggregate_reports([first, second], expected_conditions=2)
    with pytest.raises(ValueError, match="expected 3 conditions"):
        aggregator.aggregate_reports([first], expected_conditions=3)


def test_aggregate_accepts_the_v2_materialized_representation_schema(tmp_path):
    aggregator = _load_aggregator()
    report = _report(family="cosine", modality="article", k_in=70)
    report["schema_version"] = "b2-e029-context-audit.v2"
    report["candidate_text_policy"] = "materialized_per_condition"
    report["conditions"][0]["candidate_text_representation"] = {
        "source_field": "texte",
        "projection": "token_prefix",
        "tokenizer_id": "frozen/gemma",
        "tokenizer_revision": "abc",
        "token_cap": 192,
    }
    path = tmp_path / "v2.json"
    _write(path, report)

    result = aggregator.aggregate_reports([path], expected_conditions=1)

    assert result["schema_version"] == "b2-e029-context-audit-aggregate.v2"
    assert result["conditions"][0]["candidate_text_representation"]["token_cap"] == 192
