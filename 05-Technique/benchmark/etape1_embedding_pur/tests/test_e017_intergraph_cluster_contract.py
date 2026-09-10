from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/81_e017_intergraph_cluster_contract.py"
TASK_RUNNER = ROOT / "scripts/82_run_e017_lightgcn_task.py"
CV_RUNNER = ROOT / "scripts/44_run_cv_lightgcn.py"
BASE_MANIFEST = ROOT / "configs/confirmatory_campaign_grouped_v2.json"


def _load_contract():
    spec = importlib.util.spec_from_file_location("e017_cluster_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_tasks_covers_every_graph_and_seed_once():
    contract = _load_contract()
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))

    tasks = contract.build_tasks(manifest)

    expected_graphs = {row["graph_id"] for row in manifest["graphs"]}
    assert len(tasks) == 33
    assert {row["graph_id"] for row in tasks} == expected_graphs
    assert {row["seed"] for row in tasks} == {42, 43, 44}
    assert len({row["task_id"] for row in tasks}) == 33
    assert len({(row["graph_id"], row["seed"]) for row in tasks}) == 33


def test_build_tasks_freezes_the_same_lightgcn_contract():
    contract = _load_contract()
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))

    tasks = contract.build_tasks(manifest)

    assert {
        (
            row["train_k"],
            row["learning_rate"],
            row["lambda_anchor"],
            row["epochs"],
            row["negative_sampling_strategy"],
            tuple(row["selection_targets"]),
        )
        for row in tasks
    } == {(2, 0.001, 1.0, 30, "random", ("art", "jp"))}


def test_transfer_paths_are_relative_existing_inputs_without_secrets():
    contract = _load_contract()
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))

    paths = contract.build_transfer_paths(manifest)

    assert paths == sorted(set(paths))
    assert all(not Path(path).is_absolute() for path in paths)
    assert all(".env" not in path for path in paths)
    assert all("password" not in path.lower() for path in paths)
    assert all("judilibre_corpus" not in path for path in paths)
    assert manifest["datasets"]["train"]["path"] in paths
    assert manifest["datasets"]["internal_eval"]["path"] in paths
    assert manifest["folds"]["path"] in paths
    assert manifest["folds"]["metadata_path"] in paths
    assert manifest["immutable_inputs"]["legi_sqlite_for_g1"]["path"] in paths
    assert {row["matrix_path"] for row in manifest["graphs"]} <= set(paths)
    hybrid_files = {"jp_ids.npy", "article_ids.npy", "node_ids.npy", "article_codes.npy"}
    for graph in manifest["graphs"]:
        if graph["graph_id"] == "G1":
            continue
        expected = {
            f"05-Technique/benchmark/etape1_embedding_pur/data/hybrid_graphs/{graph['graph_id']}/{filename}"
            for filename in hybrid_files
        }
        assert expected <= set(paths)


def test_materialized_campaign_is_explicitly_internal_and_exploratory(tmp_path):
    contract = _load_contract()
    manifest = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))

    result = contract.materialize(manifest, tmp_path)
    campaign = json.loads((tmp_path / "campaign_manifest.json").read_text(encoding="utf-8"))
    task_rows = [
        json.loads(line)
        for line in (tmp_path / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert result["n_tasks"] == 33
    assert len(task_rows) == 33
    assert campaign["experiment_id"] == "E017"
    assert campaign["scientific_status"] == "exploratory_internal_evaluation"
    assert campaign["internal_eval_authorized"] is True
    assert campaign["source_campaign_id"] == manifest["campaign_id"]


def test_cv_runner_honors_lkg_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("LKG_REPO", str(tmp_path))

    runner = _load_script(CV_RUNNER, "cv_runner_e017_portability")

    assert runner.REPO == tmp_path


def test_task_runner_builds_isolated_cv_command(monkeypatch, tmp_path):
    monkeypatch.setenv("LKG_REPO", str(tmp_path))
    runner = _load_script(TASK_RUNNER, "e017_task_runner")
    task = {
        "task_id": 7,
        "graph_id": "G7-citation-AA-cit1-sem025-knn5",
        "seed": 43,
        "train_k": 2,
        "learning_rate": 0.001,
        "lambda_anchor": 1.0,
        "epochs": 30,
        "negative_sampling_strategy": "random",
        "selection_targets": ["art", "jp"],
    }
    config = {
        "outputs": {
            "cv_root": "relative/cv",
            "final_root": "relative/final",
        }
    }

    command = runner.build_cv_command(task, config, repo=tmp_path, python="python")
    rendered = " ".join(command)

    assert command[0] == "python"
    assert "44_run_cv_lightgcn.py" in rendered
    assert "--graph-version G7-citation-AA-cit1-sem025-knn5" in rendered
    assert "--seed 43" in rendered
    assert "--train-k 2" in rendered
    assert "--selection-target art" in rendered
    assert "--selection-target jp" in rendered
    assert str(
        tmp_path
        / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/train_augmented_retrievable_strict"
    ) in rendered
    assert str(tmp_path / "relative/cv/G7-citation-AA-cit1-sem025-knn5/seed_43/lightgcn") in rendered
    assert "eval_rich_retrievable_strict" not in rendered


def test_task_runner_rejects_wrong_array_index(tmp_path):
    runner = _load_script(TASK_RUNNER, "e017_task_runner_bounds")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"task_id": 0}) + "\n", encoding="utf-8")

    try:
        runner.load_task(tasks, 1)
    except IndexError as exc:
        assert "task index 1" in str(exc)
    else:
        raise AssertionError("out-of-range task index must fail")


def test_verify_inputs_detects_changed_file(tmp_path):
    contract = _load_contract()
    source = tmp_path / "sealed.txt"
    source.write_text("sealed", encoding="utf-8")
    campaign = {
        "inputs": {
            "sealed.txt": {
                "size_bytes": source.stat().st_size,
                "sha256": contract.sha256_file(source),
            }
        }
    }

    assert contract.verify_inputs(campaign, tmp_path) == []
    source.write_text("changed", encoding="utf-8")

    problems = contract.verify_inputs(campaign, tmp_path)
    assert len(problems) == 1
    assert problems[0]["path"] == "sealed.txt"
    assert problems[0]["problem"] in {"size_mismatch", "sha256_mismatch"}


def test_cluster_launcher_exposes_safe_staged_modes():
    launcher = ROOT / "scripts/run_e017_intergraph_cluster.sh"

    content = launcher.read_text(encoding="utf-8")

    assert 'case "$MODE" in' in content
    assert "inventory)" in content
    assert "sync)" in content
    assert "submit-cpu)" in content
    assert "rsync" in content and "--checksum" in content
    assert "ServerAliveInterval=30" in content
    assert "for attempt in 1 2 3" in content
    assert "aftercorr" in content
    assert ".env" not in content


def test_cpu_sbatch_uses_the_complete_system_python_runtime():
    config = json.loads(
        (ROOT / "configs/e017_intergraph_graded_jp_cluster.json").read_text(
            encoding="utf-8"
        )
    )
    cv_script = (ROOT / "scripts/sbatch_e017_lightgcn_cv.sh").read_text(
        encoding="utf-8"
    )
    replay_script = (
        ROOT / "scripts/sbatch_e017_lightgcn_replay.sh"
    ).read_text(encoding="utf-8")

    assert config["cluster"]["python"] == "/usr/bin/python3"
    assert "E017_PYTHON" in cv_script
    assert "E017_PYTHON" in replay_script
    assert ".venv-benchmark" not in cv_script
    assert ".venv-benchmark" not in replay_script
    assert "#SBATCH --array=0-10%10" in cv_script
    assert "#SBATCH --array=0-10%10" in replay_script
    assert "for seed_offset in 0 1 2" in cv_script
    assert "for seed_offset in 0 1 2" in replay_script
    assert "SLURM_ARRAY_TASK_ID * 3 + seed_offset" in cv_script
    assert "SLURM_ARRAY_TASK_ID * 3 + seed_offset" in replay_script
