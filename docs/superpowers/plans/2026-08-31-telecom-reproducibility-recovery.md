# Télécom Reproducibility Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` or an equivalent task-by-task workflow. Steps use checkbox syntax for tracking.

**Goal:** Audit the existing PPR final results and complete the thirteen missing E021 reranking units without changing historical artefacts.

**Architecture:** The PPR CPU job is read-only with respect to existing final rankings; it recomputes the reported top-10 metrics from stored rankings and verifies sealed campaign provenance. The E021 GPU job verifies input hashes, resumes an append-only response history by family/question, and writes new aggregate and receipt files outside the v5 result directory.

**Tech Stack:** Python 3.12, pandas/Parquet, Slurm, vLLM 0.19.1, Bash, SHA-256.

**Spec:** `experiments/confirmatory-recovery/manifest_ppr_final_audit_v1.json` and `experiments/reranking-comparable/manifest_cluster_gpu_runtime_v5_resume_v1.json`.

## Global Constraints

- Do not modify paper files, historical manifests, `_final_grouped_v2`, or the E021 v5 response history except by append-only runner output.
- PPR selection stays in `train_augmented_retrievable_strict` with five grouped folds.
- E021 uses the same questions, candidate-pool size, output size, model, prompt and temperature for cosine, PPR and LightGCN.
- Use `LKG_REPO` and `LKG_DATA_ROOT`; never require a personal local path.

### Task 1: PPR final audit

**Files:**

- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/71_audit_ppr_final_recovery.py`
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_ppr_final_audit.sh`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_ppr_final_recovery_audit.py`

- [x] Write a fixture with two PPR targets, rankings and sealed hashes.
- [x] Verify the test fails before the audit module exists.
- [x] Verify champions, summary coverage, rankings, hashes and recomputed metrics.
- [x] Reject an undeclared source-manifest hash.

### Task 2: E021 completion receipt

**Files:**

- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/72_finalize_e021_resume.py`
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_e021_reranking_resume.sh`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_e021_resume_completion.py`

- [x] Write complete and incomplete family fixtures.
- [x] Verify the test fails before the finalizer module exists.
- [x] Require every family/question key plus complete aggregate coverage.
- [x] Preserve incomplete evidence while returning a non-zero status.

### Task 3: Portable Télécom submission

**Files:**

- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/run_telecom_reproducibility.sh`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_telecom_launcher.py`

- [x] Test remote `$HOME` resolution through a fake SSH/Slurm boundary.
- [x] Reject remote tracked changes before code synchronization.
- [x] Submit named CPU-audit and GPU-resume jobs with explicit resource requests.
- [x] Run targeted tests, Bash syntax checks, JSON validation and `git diff --check`.
