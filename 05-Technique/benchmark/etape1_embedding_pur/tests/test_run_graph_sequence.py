from pathlib import Path
import importlib.util
import sys


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "48_run_graph_sequence.py"
)
spec = importlib.util.spec_from_file_location("run_graph_sequence", SCRIPT)
run_graph_sequence = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = run_graph_sequence
spec.loader.exec_module(run_graph_sequence)


def test_build_steps_uses_graph_specific_cv_root(monkeypatch, tmp_path):
    bench_dir = tmp_path / "G2" / "train_augmented_retrievable_strict"
    bench_dir.mkdir(parents=True)

    monkeypatch.setattr(
        run_graph_sequence.graph_protocol,
        "resolve_graph_bench_dir",
        lambda graph_version, split: bench_dir,
    )

    steps = run_graph_sequence.build_steps("G2")

    assert [step.name for step in steps] == [
        "b3_b4_cv",
        "ppr_cv",
        "lightgcn_cv",
        "final_champions",
    ]
    assert steps[-1].cmd[-1] == str(bench_dir / "_cv")
    assert all("--graph-version" in step.cmd for step in steps)


def test_build_steps_resolves_legacy_and_grouped_v2_roots(monkeypatch, tmp_path):
    bench_dir = tmp_path / "G2" / "train_augmented_retrievable_strict"
    bench_dir.mkdir(parents=True)
    monkeypatch.setattr(run_graph_sequence.graph_protocol, "BENCH_ROOT", tmp_path)
    monkeypatch.setattr(
        run_graph_sequence.graph_protocol,
        "resolve_graph_bench_dir",
        lambda graph_version, split: bench_dir,
    )

    legacy = run_graph_sequence.build_steps("G2")
    grouped = run_graph_sequence.build_steps("G2", protocol_version="grouped_v2")

    assert legacy[-1].cmd[-1] == str(bench_dir / "_cv")
    assert grouped[-1].cmd[-1] == str(tmp_path / "_cv_grouped_v2" / "G2")
    assert grouped[-1].expected_path == tmp_path / "_final_grouped_v2" / "G2" / "final_champions_summary.csv"


def test_wait_for_path_returns_once_target_exists(tmp_path):
    target = tmp_path / "ready.txt"
    target.write_text("ok")

    run_graph_sequence.wait_for_path(target, 0.01)

    assert target.exists()


def test_write_status_creates_parent_and_json(tmp_path, monkeypatch):
    status_path = tmp_path / "nested" / "status.json"
    monkeypatch.setattr(run_graph_sequence, "STATUS_PATH", status_path)

    run_graph_sequence.write_status({"status": "waiting", "graphs": ["G2", "G3"]})

    assert status_path.exists()
    payload = __import__("json").loads(status_path.read_text())
    assert payload["status"] == "waiting"
    assert payload["graphs"] == ["G2", "G3"]
