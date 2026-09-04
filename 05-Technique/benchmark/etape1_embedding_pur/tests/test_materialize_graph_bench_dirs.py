from pathlib import Path
import importlib.util
import json

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "47_materialize_graph_bench_dirs.py"
)
spec = importlib.util.spec_from_file_location("materialize_graph_bench_dirs", SCRIPT)
materialize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materialize)


def test_materialize_graph_split_copies_artifacts_and_local_coverage(tmp_path, monkeypatch):
    bench_root = tmp_path / "bench"
    split = "eval_rich_retrievable_strict"
    src_dir = bench_root / split
    src_dir.mkdir(parents=True)
    (src_dir / "bench_global.json").write_text(json.dumps({"questions": [{"qid": "q1"}]}))
    (src_dir / "stats.json").write_text(json.dumps({"n_questions": 1}))
    np.save(src_dir / "questions_ids.npy", np.array(["q1"], dtype=object))
    np.save(src_dir / "questions_emb.npy", np.array([[1.0, 2.0]], dtype=np.float32))

    monkeypatch.setattr(materialize.graph_protocol, "BENCH_ROOT", bench_root)

    coverage = {
        "datasets": {
            split: {
                "g1": {
                    "questions": 754,
                    "strict_q_any_pct": 100.0,
                }
            }
        }
    }

    out_dir = materialize.materialize_graph_split("G1", split, coverage, force=False)

    assert out_dir == bench_root / "G1" / split
    assert (out_dir / "bench_global.json").exists()
    assert (out_dir / "stats.json").exists()
    assert np.load(out_dir / "questions_ids.npy", allow_pickle=True).tolist() == ["q1"]
    local_coverage = json.loads((out_dir / "coverage_summary.json").read_text())
    assert local_coverage["graph_version"] == "G1"
    assert local_coverage["coverage"]["questions"] == 754


def test_materialize_g5_uses_g1_coverage_source(tmp_path, monkeypatch):
    bench_root = tmp_path / "bench"
    split = "eval_rich_retrievable_strict"
    src_dir = bench_root / split
    src_dir.mkdir(parents=True)
    (src_dir / "bench_global.json").write_text(json.dumps({"questions": [{"qid": "q1"}]}))
    (src_dir / "stats.json").write_text(json.dumps({"n_questions": 1}))
    np.save(src_dir / "questions_ids.npy", np.array(["q1"], dtype=object))
    np.save(src_dir / "questions_emb.npy", np.array([[1.0, 2.0]], dtype=np.float32))
    monkeypatch.setattr(materialize.graph_protocol, "BENCH_ROOT", bench_root)
    coverage = {"datasets": {split: {"g1": {"questions": 754}}}}

    out_dir = materialize.materialize_graph_split(
        "G5-citation-knn5",
        split,
        coverage,
        force=False,
    )

    local_coverage = json.loads((out_dir / "coverage_summary.json").read_text())
    assert local_coverage["graph_version"] == "G5-citation-knn5"
    assert local_coverage["coverage_source_graph"] == "G1"
    assert local_coverage["coverage"]["questions"] == 754
