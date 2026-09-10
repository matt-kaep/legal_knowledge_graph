# E030 retry v3 r6 — audited exploratory LLM-as-a-Judge exports

These light exports contain no question, candidate text, raw model response or ranking.

- Execution manifest SHA-256: `b4bb39dcbfff639f3c2cc5a3aecc637f2ff6475c6f1e17dcf61a91a6472746cd`.
- Evaluation: 754 A3 questions, fixed K=10 (7,540 positions per shard).
- Model: Gemma `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` at revision `4033b16200f4152e55e100ea12dc388c537df622`, temperature 0.
- Judge gains: A=1, B=0.5, C/D/E/non_jugeable=0; duplicates after the first occurrence have gain 0.

All 17 r6 retry shards are technically complete and clean. The five clean E030 v2 shards remain archived diagnostics and are not reused in these scores or means. `zero_slot` is an explicitly frozen unresolved position: no LLM call, fixed-K gain 0, never a transport error.

`judge_scores_exploratory.csv` has one score per r6 shard. `lightgcn_seed_mean_exploratory.csv` averages reranked LightGCN only after clean seeds 42, 43 and 44. These are exploratory LLM-as-a-Judge results, separate from exact retrieval metrics, and cannot support a comparative claim until `lawyer_agreement.json` exists.
