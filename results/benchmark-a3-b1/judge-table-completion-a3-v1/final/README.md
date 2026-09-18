---
type: result-export
status: exploratory
owner: assainissement
---

# LLM-as-a-Judge results on the frozen legal retrieval benchmark

All scores remain exploratory until a blind legal audit is completed.
Human agreement must be recorded in `lawyer_agreement.json` before legal
validation or comparative conclusions. Exact retrieval metrics are separate.

`judge_scores_for_paper.csv` contains direct generation, raw retrieval and
reranked top-10 rows in explicit stages. `comparison_role=reranking_input`
identifies PPR G6-AA Judicial Decisions; PPR G7-AA belongs to the main table.
These are distinct conditions and must never be substituted or averaged.
Other raw retrieval rows also serve as their own reranking baselines.

`judge_scores_by_condition.csv` and `shards/` retain each seed separately.
LightGCN means require precisely the clean seeds 42,43,44 for one task and
stage. Semicolon-separated source hashes in means follow that seed order;
they identify three rankings, never a synthetic average ranking.
`lightgcn_three_seed_means.csv` contains only means that passed this gate.
Blank scores mean pending, never zero. The zero Direct LLM decision score
comes from frozen unresolved reference slots, not completed adverse judgments.

The Judge sees only the question and full article text or decision synthese.
Model, revision, prompt hash and rubric hash are included in every row.
Gains are A=1, B=0.5, C/D/E/non_jugeable=0, with fixed K=10. Repeated
candidates after their first occurrence receive zero gain. Frozen unresolved
slots remain zero-valued positions; HTTP errors never become labels or zeros.
Judging the reranked output uses only input depth 70, output depth 10.
No Judge depth sweep, retriever training or new reranking was performed.

The public receipt hashes every result and the evidence manifest supplies
the exact remote source paths, ranking identities and ledger hashes.
