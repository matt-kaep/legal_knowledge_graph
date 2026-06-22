# Task 1 Report — Shared protocol helpers and deterministic folds

## Objective
Implement the shared benchmark protocol helpers and the deterministic 5-fold assignment generator for the doctrine_v3plus benchmark pipeline.

## Delivered

### `05-Technique/benchmark/etape1_embedding_pur/scripts/graph_protocol.py`
- Added `REPO` and `BENCH_ROOT` with the repo's absolute-path convention.
- Implemented `resolve_graph_bench_dir(graph_version, split)` to target `data/doctrine_v3plus_bench/<graph_version>/<split>`.
- Implemented `load_bench_questions(bench_dir)` to read `bench_global.json` and return the `questions` list.
- Implemented `metric_rank_tuple(row, modality)` with the required ordering:
  - JP: `hit, ndcg, mrr, m1, m2`
  - non-JP: strict-suffixed metrics when available, with fallback to base keys.

### `05-Technique/benchmark/etape1_embedding_pur/scripts/41_make_kfold_assignments.py`
- Added a deterministic fold builder using 5 folds by default.
- Grouped questions by `(n_articles_strict, n_jp_resolues)` and used a seed-derived deterministic shuffle within each group.
- Assigned folds round-robin and stabilized output ordering by `qid`.
- Added CLI support writing:
  - `fold_assignments.csv`
  - `fold_assignments_meta.json`

### Tests
- Added `tests/test_graph_protocol.py`
- Added `tests/test_kfold_assignments.py`

## Verification
Ran:

```bash
pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_graph_protocol.py 05-Technique/benchmark/etape1_embedding_pur/tests/test_kfold_assignments.py -v
```

Result: 3 passed.

## Commit
- `25749c0` — `feat: add graph protocol helpers and k-fold assignments`

## Notes
- Output paths are rooted under `data/doctrine_v3plus_bench`, not `data/global_bench`.
- The implementation preserves deterministic behavior for repeated runs with the same seed and input questions.

## Fix Follow-up — 2026-06-22

Addressed the reviewer findings without widening ownership beyond the four Task 1 files.

### What changed
- `graph_protocol.py`
  - `resolve_graph_bench_dir()` now supports both layouts:
    - future: `data/doctrine_v3plus_bench/<graph_version>/<split>/`
    - current producer layout: `data/doctrine_v3plus_bench/<split>/`
  - Added shared protocol constants/helpers for the official benchmark path:
    - `OFFICIAL_TRAIN_SPLIT = train_augmented_retrievable_strict`
    - `OFFICIAL_N_FOLDS = 5`
    - shared protocol dir under `data/doctrine_v3plus_bench/_protocol/<split>/`

- `41_make_kfold_assignments.py`
  - The CLI no longer derives folds from a graph-local bench directory.
  - Canonical folds are now always built from the official source bench:
    - `data/doctrine_v3plus_bench/train_augmented_retrievable_strict/bench_global.json`
  - Outputs are now written once to the shared protocol location:
    - `data/doctrine_v3plus_bench/_protocol/train_augmented_retrievable_strict/fold_assignments.csv`
    - `data/doctrine_v3plus_bench/_protocol/train_augmented_retrievable_strict/fold_assignments_meta.json`
  - The CLI now rejects protocol drift for the official path:
    - non-official `--split`
    - any `--n-folds` other than `5`
  - Kept `--graph-version` only as compatibility metadata; it no longer influences the canonical fold source or output path.

### Why this closes the findings
- Shared frozen reference:
  - folds are generated once from the full official train strict bench, independent of graph coverage.
- No CLI drift:
  - official generation is locked to `train_augmented_retrievable_strict` and exactly `5` folds.
- Layout compatibility:
  - graph bench resolution now works with today’s root-per-split producer layout and will prefer graph-version subdirectories once they exist.

### Added regression coverage
- `test_graph_protocol.py`
  - prefers graph-version layout when present
  - falls back to legacy root-per-split layout when graph-version layout is absent
- `test_kfold_assignments.py`
  - writes canonical folds to the shared protocol directory from the official train strict bench
  - rejects non-official split values
  - rejects non-5 fold counts

### Verification
Ran:

```bash
pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_graph_protocol.py 05-Technique/benchmark/etape1_embedding_pur/tests/test_kfold_assignments.py -v
```

Result: `8 passed`
