import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess


RESOLVER_PATH = (
    Path(__file__).resolve().parents[1]
    / "05-Technique/benchmark/etape1_embedding_pur/scripts/e029_audited_pool_resolver.py"
)
LAUNCHER_V3_PATH = (
    Path(__file__).resolve().parents[1]
    / "05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_b2_e029_prepare_jobs_v3.sh"
)


def load_resolver():
    spec = importlib.util.spec_from_file_location("e029_audited_pool_resolver", RESOLVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolves_audited_pool_from_its_declared_preflight_location(tmp_path):
    """Replacing declared audit provenance with one fixed pool root must fail this test."""
    data_root = tmp_path / "data"
    pool = data_root / "preflight_a3" / "pools" / "cosine_jp_kin50.jsonl"
    pool.parent.mkdir(parents=True)
    pool.write_text('{"qid":"q1","candidates":[]}\n', encoding="utf-8")
    audit_job_files = [
        {
            "path": str(pool),
            "sha256": hashlib.sha256(pool.read_bytes()).hexdigest(),
        }
    ]

    resolver = load_resolver()

    assert resolver.resolve_audited_pool(
        audit_job_files,
        data_root=data_root,
        family="cosine",
        modality="jp",
        k_in=50,
    ) == pool


def test_spooled_jobs_launcher_uses_lkg_repo_to_find_its_core(tmp_path):
    """Replacing LKG_REPO with dirname($0) makes Slurm-spooled launchers fail."""
    repo = tmp_path / "repo"
    core = repo / "05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_b2_e029_prepare_jobs_v1.sh"
    core.parent.mkdir(parents=True)
    core.write_text('#!/usr/bin/env bash\nprintf "core:%s\\n" "$E029_JOBS_PREPARATION_MANIFEST_FILENAME"\n', encoding="utf-8")
    spool = tmp_path / "slurm-spool"
    spool.mkdir()
    spooled_launcher = spool / "sbatch_b2_e029_prepare_jobs_v3.sh"
    shutil.copy2(LAUNCHER_V3_PATH, spooled_launcher)

    completed = subprocess.run(
        ["bash", str(spooled_launcher)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "LKG_REPO": str(repo)},
    )

    assert completed.stdout == "core:b2_reranking_comparable_a3_jobs_preparation_v3.json\n"
