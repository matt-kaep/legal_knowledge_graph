---
type: result-export
status: exploratory
owner: assainissement
---

# Reranking depth on the frozen French legal retrieval benchmark

`reranking_depth.csv` contains 42 exact results: two tasks, three retrievers,
input depths 10,20,30,40,50,60,70 and output depth 10, on 754 questions.
No model, retrieval, reranking or Judge call was made for this export.

PPR uses G6-AA for both tasks. Its Judicial Decisions baseline is
0.22822723253757737; it is distinct from G7-AA in the main retrieval table.
LightGCN uses G6 and the arithmetic mean of seeds 42,43,44, checked against
the sealed per-seed metrics. Semicolon-separated ranking hashes follow
that seed order; each identifies the full multi-depth source parquet,
filtered by task and input depth, rather than a fabricated mean ranking.
The source tidy CSV supplies the unrounded before/after values.
The original figure files are unchanged. There are no points beyond 70.

These are exploratory internal-evaluation results. `audit_receipt.json`
records the source paths, exact SHA-256 values and aggregation rules.

Suggested description: “Exact normalized hit at rank 10 before and after
LLM reranking, as a function of input-pool depth, on 754 evaluation queries.
Articles and judicial decisions are evaluated separately. LightGCN results
average three frozen replay seeds.”
