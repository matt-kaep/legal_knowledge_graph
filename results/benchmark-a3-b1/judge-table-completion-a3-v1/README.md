---
date: 2026-09-14
type: result-handoff
status: six-conditions-running
owner: assainissement
---

# Frozen legal retrieval results — table completion

## Current deliverables

The initial inventory and fresh independent audit preceded every model call.
Seventeen existing clean conditions passed: job hashes, A3 question identity,
candidate order and full source cards, exact question–rank–candidate identity,
response and materialization hashes, complete 754 × 10 coverage and an
independent recomputation of gains. Their scores are unchanged.

`available/judge_scores_for_paper.csv` is the immediately usable partial table.
It separates raw retrieval, direct generation and reranked output. Empty
values are pending, not zeros. `available/shards/` contains one CSV per
condition, including the six pending conditions. Raw LightGCN means are
withheld until seed 42 completes; the three-seed reranked means are available.

All Judge results remain exploratory until blind legal review and recorded
human agreement (`lawyer_agreement.json`). Direct LLM Judicial Decisions has
7,540 frozen unresolved slots: its zero is not 7,540 negative LLM judgments.

## Six authorized completions

| Condition | Graph | Seed | GPU job | Reserved port |
|---|---|---|---|---|
| Cosine — Judicial Decisions | not applicable | — | 992279 | 18501 |
| LightGCN — Articles | G6 | 42 | 992280 | 18502 |
| LightGCN — Judicial Decisions | G6 | 42 | 992281 | 18503 |
| PPR — Articles | G6-AA | — | 992282 | 18504 |
| PPR — Judicial Decisions, reranking input | G6-AA | — | 992283 | 18505 |
| PPR — Judicial Decisions, main table | G7-AA | — | 992284 | 18506 |

The old frozen `ppr_jp` Judge list is actually G7-AA, as verified from
`selected_graph_version` in its ranking. A new G6-AA list was projected from
the already frozen scoped replay (no new retrieval). These two conditions
are distinct. The G6-AA input is the before-reranking comparison; the G7-AA
input belongs to the main retrieval table. The existing reranked PPR outputs
remain G6-AA for both tasks.

## Frozen contract and execution

- 754 unchanged A3 questions, fixed output K=10.
- Gemma `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, revision
  `4033b16200f4152e55e100ea12dc388c537df622`, temperature 0.
- Exact clean-run prompts, rubric, source texts and unresolved-slot policy.
- Judge input: question + full Article `texte` or complete decision `synthese`.
- Gains A=1, B=0.5, others=0; repetitions after their first occurrence get 0.
- 16,384 context tokens, 256 output tokens reserved. All six lists pass the
  context audit. This is distinct from the reranker's 512-token output budget.
- Same immutable fail-closed inference program; HTTP errors never become
  judgments. Independent local ports, `/health`, `/v1/models`, exact served
  model identity and snapshot revision are checked before inference.
- Smoke job 992269 passed on L40S/node52 in 1m46s, without a judgment.
- Jobs 992279–992284 were submitted and observed running. CPU aggregation
  job 992285 has `afterok` dependencies on all six. It independently audits
  all 23 conditions, retaining seeds separately and requiring 42,43,44 before
  each of the four LightGCN means (two tasks × two stages).
- After approximately 5m40s, all six runs had produced 691–758 valid response
  rows each, with no technical-failure receipt: two A100, three L40S and one
  H100 allocations. Thread monitoring is scheduled every ten minutes through
  `finaliser-les-six-scores-judge-du-papier` and will retrieve and verify the
  final light package after the dependent audit succeeds.
- Expected remaining time at submission: approximately 45–75 minutes if no
  technical failure, based on historical 38–40-minute clean shards; not a guarantee.
  Six GPU jobs imply roughly 4–7.5 GPU-hours. No training is performed.

## Paths and SHA-256

Paths below are relative to this directory unless indicated otherwise.

| Artifact | SHA-256 | Decision |
|---|---|---|
| `preflight/judge_table_completion_manifest.json` | `a22c65803bf1cb39a9992f563746b6fb26f564e7fc1fc0a704ebab789bd404ee` | Frozen six-condition execution contract |
| `preflight/preparation_receipt.json` | `0b74088d37c720edfba489c29b9a0f2d13122ba0dc75372b51aa7a68f2c2080f` | CPU preparation passed |
| `preflight/reused_evidence_audit.json` | `53118c939f9d86cfb3dd0067a959d621bece9afc472ace94d08773f5ece333d8` | Reuse 17 unchanged scores |
| `preflight/missing_conditions.csv` | `ca16eb41b8b91a8caf6d1ad9e3b731daf82f3590ed9bd6529d8a06288a3b8cb3` | Six conditions to calculate |
| `preflight/smoke_receipt.json` | `e86576ecca38f21d4266400459552d31b14582457f6968df0ceaa85b31170e77` | Service identity passed |
| `preflight/submission_receipt.json` | `070b4c2e1a8cda93a539fe2bf82f99d1cfea78a218ebe7ec01fcba241780c855` | Six GPU jobs + dependent audit |
| `available/judge_scores_for_paper.csv` | `8c20214fdafd71fa45d078a288352c061bb26c895e175cdd8898ffa3fd093660` | Reusable descriptive scores; pending blanks |
| `available/judge_scores_by_condition.csv` | `ecd20e15c677c3d44f5572976fd47275e91f795eb6d7d4d7d34b0c7c5ae52193` | Per-seed detail, 17 complete + 6 pending |
| `available/lightgcn_three_seed_means.csv` | `9d6132ea92263656579bb8536b3f9a2049608bb1dc4ecc8abc819d0d13390ae6` | Reranked means only, seeds 42;43;44 |
| `available/audit_receipt_public.json` | `40a3e2e8800c9cd47d8b8cc48fc91415b10b56600aaf83783784796fc146736a` | Public evidence receipt |

The older technical-failure campaign remains archived; no score from it is
reused, including its five diagnostically clean shards. Historical non-A3
retrieval results are excluded. The immutable preparation source is retained
as `scripts/124_prepare_judge_tables_frozen_v1.py`; the extended audit program
has its own hash in the submission receipt and checks that hash when starting.

## Data access and finalization

Remote account: `kaeppelin-22@gpu-gw.enst.fr`.
Code root: `/home/ids/kaeppelin-22/legal_knowledge_graph_e030_v3`.
Data root: `/home/ids/kaeppelin-22/legal_knowledge_graph_b1_a3_data/05-Technique/benchmark/etape1_embedding_pur/data`.
Execution directory below data root:
`doctrine_v3plus_bench/_judge_table_completion_a3_v1_20260914`.

Expected final light package below code root:
`results/benchmark-a3-b1/judge-table-completion-a3-v1/final/`.
This directory is not a completed deliverable until the dependent audit
succeeds. Its public receipt must attest 23 complete conditions and four
complete LightGCN means. Retrieve the light package only and verify
`package_hashes.json` before making it available to the paper session.
Do not copy raw questions, full cards or response ledgers into public Git.

The authorized audit command (already scheduled after the six GPU jobs) is:

```sh
python scripts/124_complete_judge_tables.py audit \
  --repo "$LKG_REPO" --data "$LKG_DATA_ROOT" \
  --manifest-sha a22c65803bf1cb39a9992f563746b6fb26f564e7fc1fc0a704ebab789bd404ee \
  --require-complete --out /path/to/new/immutable/audit-directory
```

Do not rerun `prepare` or `126_submit_judge_table_completion.py` for this
existing run. Both refuse overwriting/duplicate submission. A technical
failure must retain its partial ledger and be diagnosed before a new attempt.

## Reranking depth data available now

Sibling directory `../reranking-depth-table-a3-v1/` contains the 42 requested
exact points, input depths 10–70 and output depth 10, with PPR G6-AA in both
tasks and LightGCN G6 averaged over seeds 42,43,44.

- `reranking_depth.csv`: `8f00ba78e88c1b222fcee9061addbee442b5fdd4cf05fb050cdfd7fb437faf1a`.
- `audit_receipt.json`: `413f450b456bf3cc6a1ba7e0148a94c9fa7f915ff96604fb3a3250d981837ba9`.
- `README.md`: `39d44b82ce2be58ea8c88fe07c820681499a958fda9eef185b2b50ed6aa94a90`.

No retrieval, reranking, training or Judge depth sweep was run to produce
this export. Original paper figures and manuscript files were not modified.
Use scientific method labels in the paper; experiment identifiers remain
internal to evidence manifests and coordination files.
