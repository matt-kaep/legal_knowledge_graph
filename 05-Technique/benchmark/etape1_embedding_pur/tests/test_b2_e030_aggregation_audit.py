from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "117_aggregate_b2_e030_audit.py"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class E030AggregationAuditTest(unittest.TestCase):
    def test_marks_any_non_ok_response_as_partial_and_blocks_lightgcn_mean(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("aggregate_e030", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / "data"; data.mkdir()
            output = data / "campaign"
            jobs_path = data / "jobs.jsonl"
            jobs = []
            for index in range(7540):
                jobs.append({"job_id": str(index), "qid": str(index // 10), "position": index % 10 + 1})
            jobs_path.write_text("".join(json.dumps(row) + "\n" for row in jobs), encoding="utf-8")
            shard_root = output / "shards" / "lightgcn-seed42-article"; materialized = shard_root / "materialized"; materialized.mkdir(parents=True)
            (shard_root / "responses.jsonl").write_text("".join(json.dumps({"job_id": str(index), "status": "ok" if index else "error"}) + "\n" for index in range(7540)), encoding="utf-8")
            positions = materialized / "per_position_judgments.jsonl"; positions.write_text("".join(json.dumps({"qid": str(index // 10)}) + "\n" for index in range(7540)), encoding="utf-8")
            questions = materialized / "per_question_judge_scores.csv"
            with questions.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["qid"]); writer.writeheader(); writer.writerows({"qid": str(index)} for index in range(754))
            scores = materialized / "judge_scores_at_10.csv"
            scores.write_text("family,modality,judge_score_at_10,questions\nlightgcn_seed42,article,0.2,754\n", encoding="utf-8")
            receipt = {"files": {path.name: digest(path) for path in (positions, questions, scores)}}
            receipt_path = materialized / "materialization_receipt.json"; receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            manifest = {"experiment_id": "E030", "outputs": {"root": "campaign"}, "shards": [{"id": "lightgcn-seed42-article", "modality": "article", "jobs": {"path": "jobs.jsonl", "sha256": digest(jobs_path), "count": 7540}}]}
            manifest_path = root / "manifest.json"; manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (shard_root / "run_receipt.json").write_text(json.dumps({"jobs_sha256": digest(jobs_path), "execution_manifest_sha256": digest(manifest_path)}), encoding="utf-8")
            artifacts = module.aggregate(manifest_path=manifest_path, data_root=data, out_dir=root / "aggregate")
            with artifacts["audit"].open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["technical_verdict"], "partial_technical")
            receipt = json.loads(artifacts["receipt"].read_text(encoding="utf-8"))
            self.assertFalse(receipt["lightgcn_seed_mean_allowed"])
