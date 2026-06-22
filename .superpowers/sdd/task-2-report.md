# Task 2 Report — Make baseline and PPR runners fold-aware

## Scope

Files changed:
- `05-Technique/benchmark/etape1_embedding_pur/scripts/26_eval_doctrine_v3plus_m1_m2.py`
- `05-Technique/benchmark/etape1_embedding_pur/scripts/25_ppr_kin_sweep.py`
- `05-Technique/benchmark/etape1_embedding_pur/scripts/42_run_cv_b3_b4.py`
- `05-Technique/benchmark/etape1_embedding_pur/scripts/43_run_cv_ppr.py`
- `05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_selection.py`

## What changed

### 1. Baseline runner (`26_eval_doctrine_v3plus_m1_m2.py`)

- Added optional `qid_filter: set[str] | None` to `eval_m1_m2(...)` so a caller can evaluate an arbitrary fold subset without mutating the canonical bench.
- Added optional `ks_in: list[int] | None` so CV wrappers can restrict the swept `k_in` values when needed.
- Kept default behavior unchanged: no filter still evaluates the full split with the historical `KS_IN`.

### 2. PPR runner (`25_ppr_kin_sweep.py`)

- Added optional `qid_filter: set[str] | None` to `load_questions(...)`.
- Threaded `qid_filter` through `main(...)` so PPR can run on a validation fold subset.
- Added `summarize_results(df)` that emits stable `method_key` rows (`PPR-sweep-k{k_in}-{variant}-a{alpha}`) for downstream CV selection.
- Preserved existing output contract under the provided `--bench-dir`.

### 3. Baseline CV wrapper (`42_run_cv_b3_b4.py`)

- New script that:
  - loads the shared canonical 5 folds from `_protocol/train_augmented_retrievable_strict/`
  - resolves the graph-aware bench directory with `graph_protocol.resolve_graph_bench_dir(...)`
  - runs `26_eval_doctrine_v3plus_m1_m2.py` on each validation fold subset in a temporary directory
  - aggregates all fold rows into `cv_results_raw.csv`
  - computes per-config means in `cv_results_summary.csv`
  - selects champions with the canonical ranking tuple (`Hit@10`, `NDCG@10`, `MRR@10`, `Recall@10`, `Normalized Rank`)
  - writes `champions.json`

### 4. PPR CV wrapper (`43_run_cv_ppr.py`)

- New script that:
  - loads the shared canonical 5 folds
  - reuses `25_ppr_kin_sweep.py` through a temporary bench directory
  - filters each run by validation `qid`
  - aggregates article-strict and JP metrics into CV summaries
  - selects champions using the same canonical tie-break order
  - writes `cv_results_raw.csv`, `cv_results_summary.csv`, and `champions.json`

### 5. Test

- Added `test_select_champion_uses_hit_then_ndcg_then_mrr()` in `test_cv_selection.py`.
- Verified the red state first: the test failed because `42_run_cv_b3_b4.py` did not exist.
- Verified the green state after implementation.

## Verification run

Commands executed:

1. `pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_selection.py -v`
   - red: failed with missing `42_run_cv_b3_b4.py`
   - green: passed

2. `python -m py_compile 05-Technique/benchmark/etape1_embedding_pur/scripts/25_ppr_kin_sweep.py 05-Technique/benchmark/etape1_embedding_pur/scripts/26_eval_doctrine_v3plus_m1_m2.py 05-Technique/benchmark/etape1_embedding_pur/scripts/42_run_cv_b3_b4.py 05-Technique/benchmark/etape1_embedding_pur/scripts/43_run_cv_ppr.py`
   - passed

3. `pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_graph_protocol.py 05-Technique/benchmark/etape1_embedding_pur/tests/test_kfold_assignments.py 05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_selection.py -v`
   - 9 passed

## Concerns

- I did not execute a full real CV benchmark run on the canonical train split because that would trigger the full embedding/PPR workloads. The implementation is verified by targeted tests and script compilation, not by an end-to-end heavy benchmark replay.

---

## Fix wave after review (2026-06-22)

Scope of this patch:
- restore historical compatibility in `05-Technique/benchmark/etape1_embedding_pur/scripts/25_ppr_kin_sweep.py`
- hard-lock CV wrappers to `train_augmented_retrievable_strict`
- publish explicit coverage counts in CV summaries/champions

What changed:
- `25_ppr_kin_sweep.py`
  - `load_questions(...)` now accepts both modern `gold_jp_ids` and historical `pourvois_cc`
  - restored historical pourvoi → JP-id resolution for legacy `data/global_bench/bench_global.json`
  - restored fallback to legacy cache names `questions_977_emb.npy` / `questions_977_ids.npy`
- `42_run_cv_b3_b4.py`
  - rejects any split other than `train_augmented_retrievable_strict` with a clear `ValueError`
  - `cv_results_summary.csv` now includes `n_questions_covered`, `n_folds_covered`, `fold_coverage`
- `43_run_cv_ppr.py`
  - same split lock
  - same explicit coverage fields in `cv_results_summary.csv`
- tests
  - extended `05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_selection.py`
  - added `05-Technique/benchmark/etape1_embedding_pur/tests/test_ppr_question_loading.py`

Verification run:
1. `pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_cv_selection.py 05-Technique/benchmark/etape1_embedding_pur/tests/test_ppr_question_loading.py -q`
   - result: `6 passed`
2. `python -m py_compile 05-Technique/benchmark/etape1_embedding_pur/scripts/25_ppr_kin_sweep.py 05-Technique/benchmark/etape1_embedding_pur/scripts/42_run_cv_b3_b4.py 05-Technique/benchmark/etape1_embedding_pur/scripts/43_run_cv_ppr.py`
   - result: passed

Concern:
- no full heavy CV benchmark replay was run in this fix wave; verification stayed focused on regression tests plus syntax compilation.
