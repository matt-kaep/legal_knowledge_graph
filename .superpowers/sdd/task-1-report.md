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
