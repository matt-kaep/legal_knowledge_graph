import importlib.util
import json
import os
from pathlib import Path

import pytest
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "64_run_confirmatory_campaign.py"
MANIFEST = ROOT / "configs" / "confirmatory_campaign_grouped_v2_repro_v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("confirmatory_campaign", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_manifest_contains_exact_approved_graph_matrix():
    payload = json.loads(MANIFEST.read_text())
    graph_ids = [graph["graph_id"] for graph in payload["graphs"]]

    assert graph_ids == [
        "G1",
        "G6-citation-AA-knn5",
        "G6-citation-JJ-knn5",
        "G7-citation-AA-cit1-sem025-knn5",
        "G7-citation-AA-cit1-sem050-knn5",
        "G7-citation-AA-cit1-sem100-knn5",
        "G7-citation-AA-cit025-sem1-knn5",
        "G7-citation-JJ-cit1-sem025-knn5",
        "G7-citation-JJ-cit1-sem050-knn5",
        "G7-citation-JJ-cit1-sem100-knn5",
        "G7-citation-JJ-cit025-sem1-knn5",
    ]
    assert len(set(graph_ids)) == 11


def test_manifest_freezes_protocol_metrics_and_grids():
    payload = json.loads(MANIFEST.read_text())

    assert payload["protocol_version"] == "grouped_v2"
    assert payload["selection"]["article"] == ["recall_at_10", "ndcg_at_10", "mrr_at_10"]
    assert payload["selection"]["jp"] == ["hit_at_10", "ndcg_at_10", "mrr_at_10"]
    ppr = payload["ppr"]
    assert len(ppr["k_in"]) * len(ppr["seed_variant"]) * len(ppr["alpha"]) == 48
    assert payload["lightgcn"]["screen"]["negative_sampling_strategy"] == ["random"]
    assert payload["lightgcn"]["robustness"]["seed"] == [42, 43, 44]


def test_preflight_verifies_real_manifest_inputs_and_hashes():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)

    report = runner.preflight(payload, verify_hashes=True)

    assert report["scientific_inputs_ok"] is True
    assert report["n_graphs"] == 11
    assert report["n_train_questions"] == 5603
    assert report["n_eval_questions"] == 754
    assert report["n_folds"] == 5
    assert report["immutable_inputs_verified"] == len(payload["immutable_inputs"])
    assert report["code_files_verified"] == len(payload["code_bundle"])
    assert report["graph_input_copies_verified"] == 84
    assert report["runtime_verified"] is True


def test_resource_assessment_blocks_when_minimum_ram_is_unmeasured():
    runner = _load_runner()
    resources = {
        "cpu_per_graph_job": 5,
        "ram_minimum_gb_per_graph_job": None,
        "ram_observed_upper_bound_gb_per_graph_job": 45,
        "max_parallel_graph_jobs": 2,
    }

    assessment = runner.assess_resources(resources, cpu_available=8, ram_available_bytes=16 * 1024**3)

    assert assessment["compatible"] is False
    assert assessment["insufficient"] == ["ram_minimum_unmeasured"]
    assert assessment["max_safe_parallel_jobs"] == 0


def test_resource_assessment_caps_parallelism_from_measured_minimum():
    runner = _load_runner()
    resources = {
        "cpu_per_graph_job": 5,
        "ram_minimum_gb_per_graph_job": 8,
        "ram_observed_upper_bound_gb_per_graph_job": 45,
        "max_parallel_graph_jobs": 2,
    }

    assessment = runner.assess_resources(resources, cpu_available=8, ram_available_bytes=16 * 1024**3)

    assert assessment["compatible"] is True
    assert assessment["max_safe_parallel_jobs"] == 1


def test_resource_assessment_uses_measured_ppr_profile_without_weakening_lightgcn():
    runner = _load_runner()
    resources = {
        "cpu_per_graph_job": 5,
        "cpu_per_ppr_job": 4,
        "ram_minimum_gb_per_graph_job": 9.1,
        "ram_minimum_gb_per_ppr_job": 3.5,
        "ram_observed_upper_bound_gb_per_graph_job": 45,
        "max_parallel_graph_jobs": 2,
    }
    available = 8 * 1024**3

    ppr = runner.assess_resources(
        resources,
        stage="ppr-cv",
        cpu_available=8,
        ram_available_bytes=available,
    )
    lightgcn = runner.assess_resources(
        resources,
        stage="lightgcn-screen",
        cpu_available=8,
        ram_available_bytes=available,
    )

    assert ppr["compatible"] is True
    assert ppr["cpu_required_per_job"] == 4
    assert ppr["ram_required_bytes_per_job"] == int(3.5 * 1024**3)
    assert ppr["max_safe_parallel_jobs"] == 2
    assert lightgcn["compatible"] is False
    assert lightgcn["insufficient"] == ["ram"]
    assert lightgcn["ram_required_bytes_per_job"] == int(9.1 * 1024**3)


def test_preflight_rejects_unsealed_question_cache(tmp_path):
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)
    cache = tmp_path / "questions_ids.npy"
    cache.write_bytes(b"changed")
    payload["immutable_inputs"] = {
        "train_question_ids": {"path": str(cache), "sha256": "0" * 64}
    }

    with pytest.raises(ValueError, match="sha256 mismatch"):
        runner.preflight(payload, verify_hashes=True)


def test_preflight_rejects_changed_fold_metadata_hash():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)
    payload["folds"]["metadata_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="sha256 mismatch"):
        runner.preflight(payload, verify_hashes=True)


def test_commands_never_write_to_legacy_namespaces():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)

    commands = runner.build_stage_commands(payload, "ppr-cv", graph_id="G1")
    rendered = "\n".join(" ".join(command) for command in commands)

    assert "_cv_grouped_v2" in rendered
    assert "/_cv/" not in rendered
    assert "_final_champions" not in rendered


@pytest.mark.parametrize("stage,dirname", [
    ("ppr-cv", "ppr"),
    ("lightgcn-screen", "lightgcn_screen"),
    ("lightgcn-tune", "lightgcn"),
])
def test_fold_metrics_are_owned_hashed_stage_artifacts(stage, dirname, tmp_path):
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)
    if stage == "lightgcn-tune":
        shortlist = tmp_path / "shortlist.json"
        shortlist.write_text(json.dumps({
            "manifest_sha256": runner.manifest_sha256(payload),
            "graph_ids": ["G1"],
            "sources": [],
        }))
    else:
        shortlist = runner.SHORTLIST_PATH

    artifacts = runner.expected_artifacts(payload, stage, graph_id="G1", shortlist_file=shortlist)

    assert any(path.as_posix().endswith(f"/G1/{dirname}/fold_metrics.csv") for path in artifacts)


def test_ppr_command_materializes_the_manifest_grid():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)

    command = runner.build_stage_commands(payload, "ppr-cv", graph_id="G1")[0]
    configs = [command[index + 1] for index, value in enumerate(command) if value == "--config"]

    assert len(configs) == 48
    assert "5:art_only:0.5" in configs
    assert "50:both:0.95" in configs


def test_internal_replay_only_requests_lightgcn_for_shortlisted_graphs(tmp_path, monkeypatch):
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)
    shortlist_path = tmp_path / "lightgcn_shortlist.json"
    shortlist_path.write_text(json.dumps({
        "manifest_sha256": runner.manifest_sha256(payload),
        "graph_ids": ["G1"],
    }))
    monkeypatch.setattr(runner, "SHORTLIST_PATH", shortlist_path)

    commands = runner.build_stage_commands(payload, "internal-replay", shortlist_file=shortlist_path)
    by_graph = {command[command.index("--graph-version") + 1]: command for command in commands}

    assert by_graph["G1"][by_graph["G1"].index("--families") + 1] == "b3_b4,ppr,lightgcn"
    assert by_graph["G6-citation-AA-knn5"][by_graph["G6-citation-AA-knn5"].index("--families") + 1] == "ppr"
    assert all("--authorize-internal-eval" in command for command in commands)


def test_shared_cosine_control_is_a_real_g1_only_stage():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)

    commands = runner.build_stage_commands(payload, "cosine-control-cv")

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--graph-version") + 1] == "G1"
    assert command[command.index("--out-dir") + 1].endswith("/G1/b3_b4")
    assert "--direct-cosine-only" in command


def test_manifest_rejects_output_escape(tmp_path):
    runner = _load_runner()
    payload = json.loads(MANIFEST.read_text())
    payload["outputs"]["cv_root"] = "../../outside"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="outputs.cv_root"):
        runner.load_manifest(path)


def test_manifest_rejects_changed_confirmatory_grid(tmp_path):
    runner = _load_runner()
    payload = json.loads(MANIFEST.read_text())
    payload["ppr"]["alpha"] = [0.85]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="ppr grid"):
        runner.load_manifest(path)


def test_manifest_accepts_a_future_positive_measured_ram_minimum(tmp_path):
    runner = _load_runner()
    payload = json.loads(MANIFEST.read_text())
    payload["resources"]["ram_minimum_gb_per_graph_job"] = 8
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    loaded = runner.load_manifest(path)

    assert loaded["resources"]["ram_minimum_gb_per_graph_job"] == 8


def test_global_status_can_satisfy_one_graph_dependency(tmp_path):
    runner = _load_runner()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x\n1\n")
    b.write_text("x\n2\n")
    status = {
        "status": "complete",
        "manifest_sha256": "manifest",
        "artifacts": [
            {"path": str(a), "sha256": runner.sha256_file(a)},
            {"path": str(b), "sha256": runner.sha256_file(b)},
        ],
    }

    assert runner.can_resume(status, "manifest", {str(a)}, allow_artifact_superset=True)


def test_status_records_stage_timing_and_dependencies(tmp_path):
    runner = _load_runner()
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("x\n1\n")

    status = runner.write_stage_status(
        tmp_path / "status.json",
        stage="ppr-cv",
        manifest_hash="abc",
        results=[{"status": "complete", "command": ["python"], "started_at": "s", "finished_at": "f"}],
        artifacts=[artifact],
        dependencies=["cosine-control-cv--G1"],
        started_at="s",
        finished_at="f",
    )

    assert status["started_at"] == "s"
    assert status["finished_at"] == "f"
    assert status["dependencies"] == ["cosine-control-cv--G1"]


def test_serialized_lightgcn_dependencies_match_enforced_requirements(tmp_path):
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)
    shortlist_path = tmp_path / "shortlist.json"
    shortlist_path.write_text(json.dumps({
        "manifest_sha256": runner.manifest_sha256(payload),
        "graph_ids": ["G1", "G7-citation-AA-cit1-sem100-knn5"],
        "sources": [],
    }))

    assert runner.verified_dependency_labels(payload, "lightgcn-shortlist", None, shortlist_path) == [
        f"lightgcn-screen--{graph['graph_id']}" for graph in payload["graphs"]
    ]
    assert runner.verified_dependency_labels(payload, "lightgcn-tune", "G7-citation-AA-cit1-sem100-knn5", shortlist_path) == ["lightgcn-tune--G1"]
    assert runner.verified_dependency_labels(payload, "lightgcn-seeds", "G1", shortlist_path) == ["lightgcn-tune--G1"]
    assert runner.verified_dependency_labels(payload, "freeze-epochs", "G1", shortlist_path) == ["lightgcn-seeds--G1"]


def test_internal_eval_stage_requires_exact_campaign_authorization():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)

    with pytest.raises(ValueError, match="authorization"):
        runner.validate_internal_eval_authorization(payload, None)
    runner.validate_internal_eval_authorization(payload, payload["campaign_id"])


def test_stage_lock_rejects_concurrent_owner(tmp_path):
    runner = _load_runner()
    lock_path = tmp_path / "ppr-cv--G1.lock"

    with runner.stage_lock(lock_path, max_parallel_jobs=2):
        with pytest.raises(RuntimeError, match="already running"):
            with runner.stage_lock(lock_path, max_parallel_jobs=2):
                pass


def test_stale_lock_recovery_archives_dead_local_owner(tmp_path):
    runner = _load_runner()
    locks = tmp_path / "locks"
    locks.mkdir()
    stale = locks / "ppr-cv--G1.lock"
    stale.write_text(json.dumps({"pid": 99999999, "hostname": runner.socket.gethostname(), "started_at": "old"}))

    proof = runner.recover_stale_locks(locks)

    assert not stale.exists()
    assert len(proof["recovered"]) == 1
    assert Path(proof["recovered"][0]["archived_path"]).is_file()


def test_stale_lock_recovery_preserves_live_owner(tmp_path):
    runner = _load_runner()
    locks = tmp_path / "locks"
    locks.mkdir()
    live = locks / "ppr-cv--G1.lock"
    live.write_text(json.dumps({"pid": os.getpid(), "hostname": runner.socket.gethostname(), "started_at": "now"}))

    proof = runner.recover_stale_locks(locks)

    assert live.exists()
    assert proof["recovered"] == []
    assert proof["retained"][0]["reason"] == "owner_alive"


def test_per_graph_stage_rejects_all_invocation():
    runner = _load_runner()

    with pytest.raises(ValueError, match="requires --graph-id"):
        runner.validate_stage_invocation("ppr-cv", None)
    runner.validate_stage_invocation("ppr-cv", "G1")


def test_existing_unverified_artifact_is_never_overwritten(tmp_path):
    runner = _load_runner()
    artifact = tmp_path / "summary.csv"
    artifact.write_text("old\n")

    with pytest.raises(FileExistsError, match="unverified campaign artifact"):
        runner.refuse_unverified_overwrite([artifact], status_path=tmp_path / "missing.json")


def test_seed_stage_does_not_treat_frozen_input_champions_as_owned_output(tmp_path):
    runner = _load_runner()
    champions = tmp_path / "champions.json"
    summary = tmp_path / "summary.csv"

    owned = runner.stage_owned_artifacts("lightgcn-seeds", [champions, summary])

    assert owned == [summary]


def test_resume_quarantines_partial_artifacts_recoverably(tmp_path):
    runner = _load_runner()
    artifact = tmp_path / "summary.csv"
    artifact.write_text("partial\n")

    proof = runner.quarantine_unverified_artifacts(
        [artifact],
        quarantine_root=tmp_path / "quarantine",
        stage="ppr-cv",
        graph_id="G1",
    )

    assert not artifact.exists()
    assert Path(proof["moves"][0]["quarantined_path"]).read_text() == "partial\n"
    assert (Path(proof["quarantine_dir"]) / "quarantine_manifest.json").is_file()


def test_resume_quarantines_the_whole_owned_output_directory(tmp_path):
    runner = _load_runner()
    output_dir = tmp_path / "ppr"
    output_dir.mkdir()
    (output_dir / "raw.csv").write_text("partial raw\n")
    (output_dir / "progress.json").write_text("{}")

    proof = runner.quarantine_unverified_artifacts(
        [output_dir],
        quarantine_root=tmp_path / "quarantine",
        stage="ppr-cv",
        graph_id="G1",
    )

    assert not output_dir.exists()
    moved = Path(proof["moves"][0]["quarantined_path"])
    assert (moved / "raw.csv").read_text() == "partial raw\n"
    assert {row["relative_path"] for row in proof["moves"][0]["files"]} == {"progress.json", "raw.csv"}


def test_dry_run_never_invokes_subprocess(monkeypatch):
    runner = _load_runner()
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: calls.append(args))

    result = runner.execute_commands([["python", "forbidden.py"]], dry_run=True)

    assert calls == []
    assert result == [{"command": ["python", "forbidden.py"], "status": "dry_run"}]


def test_resume_requires_matching_manifest_and_artifact_hashes(tmp_path):
    runner = _load_runner()
    artifact = tmp_path / "summary.csv"
    artifact.write_text("value\n1\n")
    status = {
        "status": "complete",
        "manifest_sha256": "manifest-hash",
        "artifacts": [{"path": str(artifact), "sha256": runner.sha256_file(artifact)}],
    }

    assert runner.can_resume(status, "manifest-hash") is True
    assert runner.can_resume(status, "manifest-hash", {str(artifact)}) is True
    assert runner.can_resume(status, "manifest-hash", {str(artifact), str(tmp_path / "new.csv")}) is False
    artifact.write_text("value\n2\n")
    assert runner.can_resume(status, "manifest-hash") is False
    assert runner.can_resume(status, "different-manifest") is False


def test_unknown_stage_is_rejected():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)

    with pytest.raises(ValueError, match="Unsupported stage"):
        runner.build_stage_commands(payload, "full-benchmark")


def test_tune_without_graph_id_consumes_frozen_shortlist(tmp_path, monkeypatch):
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)
    shortlist_path = tmp_path / "lightgcn_shortlist.json"
    shortlist_path.write_text(json.dumps({
        "manifest_sha256": runner.manifest_sha256(payload),
        "graph_ids": ["G1", "G7-citation-AA-cit1-sem100-knn5"],
    }))
    monkeypatch.setattr(runner, "SHORTLIST_PATH", shortlist_path)

    commands = runner.build_stage_commands(payload, "lightgcn-tune", shortlist_file=shortlist_path)

    assert len(commands) == 2
    assert all("/lightgcn" in command[command.index("--out-dir") + 1] for command in commands)
    assert all("lightgcn_screen" not in " ".join(command) for command in commands)


def test_screen_uses_isolated_namespace():
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)

    command = runner.build_stage_commands(payload, "lightgcn-screen", graph_id="G1")[0]

    assert command[command.index("--out-dir") + 1].endswith("/G1/lightgcn_screen")


def test_atomic_status_round_trip_and_resume(tmp_path):
    runner = _load_runner()
    artifact = tmp_path / "artifact.csv"
    artifact.write_text("x\n1\n")
    status_path = tmp_path / "status.json"

    status = runner.write_stage_status(
        status_path,
        stage="ppr-cv",
        manifest_hash="abc",
        results=[{"status": "complete", "command": ["python"]}],
        artifacts=[artifact],
    )

    assert json.loads(status_path.read_text()) == status
    assert runner.can_resume(status, "abc") is True


def test_freeze_dry_run_never_calls_writer(tmp_path, monkeypatch):
    runner = _load_runner()
    payload = runner.load_manifest(MANIFEST)
    shortlist_path = tmp_path / "lightgcn_shortlist.json"
    shortlist_path.write_text(json.dumps({
        "manifest_sha256": runner.manifest_sha256(payload),
        "graph_ids": ["G1"],
        "sources": [],
    }))
    monkeypatch.setattr(runner, "campaign_shortlist_path", lambda _payload: shortlist_path)
    monkeypatch.setattr(
        runner,
        "freeze_lightgcn_champions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("writer called")),
    )

    assert runner.main(["--manifest", str(MANIFEST), "--stage", "freeze-epochs", "--graph-id", "G1", "--dry-run"]) == 0


def test_freeze_rejects_robustness_proof_from_another_champion(tmp_path):
    runner = _load_runner()
    cv_root = tmp_path / "cv"
    lightgcn_dir = cv_root / "G1" / "lightgcn"
    robustness_dir = cv_root / "G1" / "lightgcn_robustness"
    lightgcn_dir.mkdir(parents=True)
    robustness_dir.mkdir(parents=True)
    champion = {
        "variant": "trained_K2", "train_k": 2, "lr": 0.001, "epochs": 30,
        "lambda_anchor": 1.0, "negative_sampling_strategy": "random",
        "selected_epoch_index": 6, "replay_epochs": 7,
    }
    champions_path = lightgcn_dir / "champions.json"
    champions_path.write_text(json.dumps({"art": champion, "jp": {**champion, "variant": "cosine_propagated_K2"}}))
    pd.DataFrame([{
        "target": "art", "champions_sha256": "wrong", "variant": "trained_K2",
        "train_k": 2, "lr": 0.001, "epochs": 30, "lambda_anchor": 1.0,
        "negative_sampling_strategy": "random", "replay_epochs": 7,
    }]).to_csv(robustness_dir / "summary.csv", index=False)
    payload = {"outputs": {"cv_root": str(cv_root)}}

    with pytest.raises(ValueError, match="robustness proof mismatch"):
        runner.freeze_lightgcn_champions(payload, ["G1"])
