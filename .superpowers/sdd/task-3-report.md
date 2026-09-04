---
task: 3
title: Make LightGCN fold-aware and export training history
date: 2026-06-22
status: done_with_concerns
---

# Task 3 Report

## Scope delivered

- Added epoch-level LightGCN training history export in [`32_lightgcn_strict.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts/32_lightgcn_strict.py).
- Added a fold-aware CV wrapper in [`44_run_cv_lightgcn.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts/44_run_cv_lightgcn.py) aligned with the shared protocol helpers and canonical folds.
- Added the protocol-figure helper in [`46_build_protocol_figures.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts/46_build_protocol_figures.py).
- Added the requested test in [`test_protocol_figures.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/tests/test_protocol_figures.py).

## What changed

### 1. `32_lightgcn_strict.py`

- `main()` now accepts `argv`, so the script can be wrapped programmatically by the CV runner.
- Added `--graph-version` to annotate exported history rows.
- Added `summarize_eval_rows()` to compute validation aggregates from per-question eval rows.
- Training now records one history row per epoch with:
  - `epoch`
  - `graph_version`
  - `variant`
  - `train_loss`
  - `bpr_loss`
  - `anchor_loss`
  - `val_hit`
  - `val_ndcg`
  - `val_mrr`
  - `val_recall`
  - `val_norm_rank`
  - `val_hit_jp`
  - `val_ndcg_jp`
- When training runs, the script now writes `lightgcn_history_<suffix>.csv` next to the existing eval and summary artifacts.

### 2. `44_run_cv_lightgcn.py`

- Enforces the official split `train_augmented_retrievable_strict`.
- Loads the shared canonical 5 folds via `graph_protocol.resolve_shared_fold_paths()`.
- Validates fold coverage against the target benchmark qids.
- Builds temporary fold-specific train/val bench subsets from:
  - `bench_global.json`
  - `questions_ids.npy`
  - `questions_emb.npy`
- Wraps `32_lightgcn_strict.py` for:
  - one baseline pass per fold (`untrained_K1/2/3`, excluding `cosine_raw` from LightGCN family selection)
  - trained LightGCN sweeps over `train_k`, `seed`, `lr`, `epochs`, `lambda_anchor`
- Produces:
  - `cv_results_raw.csv`
  - `cv_results_summary.csv`
  - `champions.json`
  - per-run `lightgcn_history_<suffix>.csv`
  - aggregate `lightgcn_history_all.csv`
- Champion selection uses the shared `metric_rank_tuple()` ordering:
  - `Hit@10`
  - `NDCG@10`
  - `MRR@10`
  - `Recall@10`
  - `Normalized Rank`

### 3. `46_build_protocol_figures.py`

- Added `prepare_lightgcn_history(df)` which converts wide LightGCN history rows to long plot-ready format:
  - output columns: `epoch`, `series`, `value`, `graph_version`, `variant`

## Validation run

### Red/green test

- Initial run of `test_protocol_figures.py` failed because `46_build_protocol_figures.py` did not exist.
- After implementation, the test passed.

### Targeted pytest

Passed:

- `05-Technique/benchmark/etape1_embedding_pur/tests/test_protocol_figures.py`
- `05-Technique/benchmark/etape1_embedding_pur/tests/test_graph_protocol.py`
- `05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_selection.py`

### Runtime smoke

Executed a real LightGCN smoke run on the strict pipeline:

- `32_lightgcn_strict.py --limit-train 5 --limit-eval 3 --epochs 1 --trained-only --output-suffix smoke_task3 --graph-version canonical`

Observed:

- `lightgcn_eval_smoke_task3.csv`
- `lightgcn_summary_smoke_task3.json`
- `lightgcn_history_smoke_task3.csv`

These smoke artifacts were deleted after verification.

## Concerns

1. `44_run_cv_lightgcn.py` has CLI/import validation but no dedicated end-to-end pytest yet.
2. I did not run the full 5-fold LightGCN CV loop on real data in this task turn, because even a minimal real sweep is materially heavier than the targeted smoke already executed on `32_lightgcn_strict.py`.

## Verdict

Implementation is complete for Task 3, with one residual confidence gap on the full end-to-end CV wrapper execution path.

---

## Fix wave addendum — 2026-06-22

Base commit reviewed: `7bf1bc818ed9451c881c8693d8776e9993905b3a`

### Findings addressed

1. Added dedicated focused tests for the risky LightGCN CV wrapper logic in [`44_run_cv_lightgcn.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts/44_run_cv_lightgcn.py):
   - fold validation rejects duplicate qids
   - fold validation rejects missing/extra qids
   - subset bench construction preserves qid/embedding alignment for train/val splits
   - champion selection respects the shared metric priority
   - non-official split is rejected at entrypoint level
2. Aligned [`46_build_protocol_figures.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts/46_build_protocol_figures.py) with the real LightGCN history schema exported by [`32_lightgcn_strict.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts/32_lightgcn_strict.py).
   - `prepare_lightgcn_history()` now melts the stable ordered set of useful exported series:
     `train_loss`, `bpr_loss`, `anchor_loss`, `val_hit`, `val_ndcg`, `val_mrr`, `val_recall`, `val_norm_rank`, `val_hit_jp`, `val_ndcg_jp`

### Files changed in this fix wave

- Added [`test_cv_lightgcn.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_lightgcn.py)
- Updated [`test_protocol_figures.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/tests/test_protocol_figures.py)
- Updated [`46_build_protocol_figures.py`](/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts/46_build_protocol_figures.py)

### Verification executed

- `pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_protocol_figures.py 05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_lightgcn.py -v`
  - result: `7 passed`
- `python -m py_compile 05-Technique/benchmark/etape1_embedding_pur/scripts/32_lightgcn_strict.py 05-Technique/benchmark/etape1_embedding_pur/scripts/44_run_cv_lightgcn.py 05-Technique/benchmark/etape1_embedding_pur/scripts/46_build_protocol_figures.py`
  - result: success

### Residual concerns

- No full real-data 5-fold LightGCN sweep was run in this fix wave; confidence is provided by focused unit tests plus syntax verification.
