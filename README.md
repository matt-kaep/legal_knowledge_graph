# Legal Knowledge Graph — reproducibility surface

This branch contains the code, manifests, tests, prompts, and lightweight exports
needed to audit and reproduce the benchmark used by the ECIR 2027 work. The legal
corpus, embeddings, and generated rankings are not committed: the local data checkout
is about 14 GiB and redistribution rights have not been cleared.

The scientific contract is `grouped_v2`: tuning uses only
`train_augmented_retrievable_strict` (5 shared grouped folds, 5,603 questions), while
`eval_rich_retrievable_strict` contains 754 questions and is an already consulted
internal evaluation, not a final lockbox. Article and JP metrics are separate. For JP,
official `Hit@10` is the coverage metric from `scripts/metrics.py`; it is not the
binary diagnostic `exact_any_gold_at_10`.

## Prerequisites and data

Use Python 3.12+ with the packages in the project environment (`numpy`, `pandas`,
`scipy`, `torch`, `pyarrow`, `pytest`). Keep code and data separate when using a
worktree:

```bash
export LKG_REPO="$PWD"
export LKG_DATA_ROOT="/path/to/legal_knowledge_graph"
export LKG_PYTHON="$LKG_DATA_ROOT/05-Technique/benchmark/etape1_embedding_pur/.venv/bin/python"
```

`LKG_DATA_ROOT` must contain the ignored
`05-Technique/benchmark/etape1_embedding_pur/data/` tree. Acquire or reconstruct inputs
according to `results/benchmark-repro-v1/data-manifest.json`, then verify the hashes
before running anything. Do not publish benchmark text, decisions, private audit keys,
embeddings, provider credentials, or raw judge responses without a separate rights review.

## Reproduction commands

Validate the portable campaign manifest and the resource gate first:

```bash
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py \
  --manifest 05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json \
  --stage preflight
```

The orchestrator supports the ordered stages `cosine-control-cv`, `ppr-cv`,
`lightgcn-screen`, `lightgcn-shortlist`, `lightgcn-tune`, `lightgcn-seeds`,
`freeze-epochs`, `internal-replay`, `diagnostics`, and `paper-exports`. Use
`--resume` for an interrupted stage and the explicit internal-evaluation authorization
only after the train-only selection artifacts are sealed. The final replay consumes
frozen epochs; it must not select them from the evaluation split.

For a read-only audit and export of the artifacts already present:

```bash
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/90_audit_and_export_reproducibility.py \
  --manifest 05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json \
  --output-dir results/benchmark-repro-v1
```

This writes the exact Article table, exact JP table, separate LLM-as-a-Judge table,
E016 exact-metric context, train-only PPR CV table, the audit JSON, a manifest snapshot,
and the data manifest. It does not launch computation. The recovered E017 judge output
is evaluated separately from exact metrics by
`scripts/84_summarize_e017_intergraph_graded_jp.py`. E016 preparation/execution and the
human gate are managed by scripts 75–80; the lawyer packet is documented under
`results/audit/e016-lawyer-audit/`.

## Verification

Run the focused reproducibility tests, then the complete suite in an environment where
`LKG_DATA_ROOT` points to the verified data checkout:

```bash
"$LKG_PYTHON" -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_paths.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_reproducibility_exports.py -q
"$LKG_PYTHON" -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests -q
```

The isolated branch must not contain changes under `07-Redaction/` or any paper TeX.

## Resource and cost notes

The sealed campaign records approximately 3.5 GiB RAM per PPR job and 9.1 GiB per
LightGCN graph job, with LightGCN restricted to one safe job under the measured profile.
The preflight refuses to run when those requirements are not met. Existing E016/E017
artifacts are therefore audited and exported rather than recomputed. E021 (the prepared
comparable reranking protocol) would require at least 2,262 provider calls: 754 questions
× 3 candidate families, with `K_in=20` and `K_out=10`. Its monetary/token cost is not
claimed until the provider model and tokenizer are frozen.

## Scientific status

- PPR grouped-v2 train-only CV: complete for 11/11 graphs, with full 5-fold and 5,603-question coverage.
- E017 LightGCN replays: operationally complete, 33/33 replays (11 graphs × 3 seeds), 754 questions and ranks 1–10 per replay; scientific status remains exploratory because the evaluation is internal and the E016 lawyer audit is incomplete.
- E016 LLM-as-a-Judge: hashes and aggregates recoverable; the 100-case lawyer audit packet exists, but `lawyer_agreement.json` is missing.
- `_final_grouped_v2`: missing; no PPR internal-evaluation replay was rerun under the local resource gate.
- Comparable reranking E021: protocol prepared, not executed; no reranking score may be cited.

The current campaign manifest is
`05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json`.
The historical manifest remains archived and immutable.
