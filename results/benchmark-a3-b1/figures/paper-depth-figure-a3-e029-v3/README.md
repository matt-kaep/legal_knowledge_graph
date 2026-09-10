# A3/E029 depth figure

Retrieval and reranking depth on the frozen A3 internal evaluation (754 questions). Top panels report exact NHit@K from frozen top-100 rankings. Bottom panels report final NHit@10 after E029 reranking at K_out=10; solid curves are reranked results and dashed lines are the corresponding retriever NHit@10. The jurisprudence PPR reranking curve uses G6-AA frozen pools and is distinct from the PPR G7-AA retrieval line in the main table. E029 is exploratory.

## Scope

- Evaluation: frozen A3 internal evaluation, 754 questions.
- Retrieval panels: exact NHit@K for K=1..100 from frozen top-100 rankings.
- Reranking panels: E029 output-512, K_in=10,20,30,40,50,60,70 and K_out=10.
- LightGCN: mean of frozen replay seeds 42, 43 and 44; seed-level E029 inputs are verified before aggregation.
- Articles: BGE-M3 cosine, PPR G6-AA and LightGCN G6. Judicial Decisions retrieval: BGE-M3 cosine, PPR G7-AA and LightGCN G6.
- Judicial Decisions reranking: BGE-M3 cosine, PPR G6-AA and LightGCN G6. The PPR G6-AA baseline is 0.22822723253757737; this is not the PPR G7-AA main-table row.

## Provenance and status

- Canonical source contract: 05-Technique/benchmark/etape1_embedding_pur/configs/paper_depth_figure_a3_e029_v3.json (SHA-256 `92d42439d827217a2700d5c6ab6565e49e467be0eecee4b2901c5a7a58801ed9`).
- All values are derived from frozen, hash-verified aggregate artifacts. No model, retrieval, training, selection, or E030 job is run by this script.
- Retrieval is A3 internal evaluation after train/CV freeze. E029 reranking is exploratory and must not support a confirmatory superiority claim.
