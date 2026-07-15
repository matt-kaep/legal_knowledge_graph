---
date: 2026-07-15
type: decision
status: approved
tags: [decision, graph, g5, lightgcn, ppr, negative-mining, benchmark]
---

# G5 typed graph optimization design

## Objective

Turn the exploratory G5 union into a controlled hybrid graph where citation and embedding relations remain complementary without one relation family dominating propagation. Select graph and training hyperparameters on train folds only, then evaluate the retained configuration once on `eval_rich_retrievable_strict`.

## Evidence driving the design

- `G5-citation-knn5 + LightGCN random` improves strict Article Hit@10 from `0.535` on G1 to `0.571`, but JP Hit@10 only from `0.237` to `0.242`.
- Increasing embedding density from KNN5 to KNN10 lowers both scores.
- After LightGCN normalization, embedding-only edges carry about `74%` of total message mass on KNN5 and `81%` on KNN10.
- About `90.5%` of G4 embedding edges are JP-JP. For a median JP, embedding edges carry `91%` of normalized message mass on KNN5.
- G5-KNN5 helps questions whose expected articles have low citation degree, but hurts questions whose expected articles have mean degree above `100`.
- On G5-KNN5, `14.5%` of cosine hard negatives are directly linked to a gold article in the embedding graph, versus `0.19%` for random negatives. The current hard-negative objective therefore conflicts with graph propagation.

## Approaches considered

### A. Keep one adjacency and tune one embedding scalar

This is the smallest change, but it cannot balance articles and JP simultaneously. A scalar near `0.01-0.02` is needed to stop JP-JP edges from dominating, while Article-Article edges need a materially larger contribution to help sparse articles. This approach remains a compatibility baseline only.

### B. Typed, separately normalized relation channels

This is the selected approach. Store and propagate three channels independently:

1. citation Article-JP;
2. semantic Article-Article;
3. semantic JP-JP.

Do not add semantic Article-JP edges in the first wave. Normalize each channel before mixing it, so relation budgets represent actual propagation budgets rather than raw edge volume. Build semantic neighbors separately by node type and compare union KNN with mutual KNN.

### C. Replace LightGCN with a heterogeneous GNN

An R-GCN or relation-aware attention model could learn relation transformations, but it adds many parameters before the useful relation channels have been identified. Defer it until typed LightGCN ablations show which channels carry useful signal.

## Graph artifacts and naming

Each typed graph artifact contains:

- `graph_citation.npz`;
- `graph_article_article.npz`;
- `graph_jp_jp.npz`;
- node identifiers, types, embedding-availability masks and metadata;
- no semantic Article-JP channel in the initial implementation.

Machine labels stay explicit:

- `G5-typed-g1-aa-k5`;
- `G5-typed-g1-jj-k3`;
- `G5-typed-g1-aa-k5-jj-k3`;
- `G5-typed-g1-aa-k5-jj-k3-mutual`;
- the retained construction is then rebuilt on G3 with prefix `G5-typed-g3-...`.

Human-facing labels spell out the method, for example `G1 citation + Article-Article semantic KNN5`.

## Propagation contracts

### PPR

Row-normalize each relation channel independently. For each source node, mix only active channels and renormalize their configured budgets. Initial citation budgets are `{0.70, 0.85, 0.95}`; the remaining budget goes to the same-type semantic channel.

### LightGCN

Symmetrically normalize each relation channel independently. One propagation layer is:

`X_next = beta_citation * A_citation * X + beta_article * A_article_article * X + beta_jp * A_jp_jp * X`.

For article rows, `beta_citation + beta_article = 1`. For JP rows, `beta_citation + beta_jp = 1`. Initial semantic budgets are `{0.05, 0.15, 0.30}`. Keep the residual average over layer 0 through layer K, as in the existing LightGCN implementation.

## Ablation sequence

### Wave 1: isolate relation value on G1

Use fixed KNN sizes `K_article=5` and `K_JP=3`, then compare:

- citation only;
- citation + Article-Article;
- citation + JP-JP;
- citation + both semantic channels;
- union KNN versus mutual KNN for the best channel combination.

Run PPR and untrained propagation first. Train LightGCN only on variants that do not regress both modalities.

### Wave 2: tune the retained typed graph

- `K_article in {3, 5, 10}`;
- `K_JP in {1, 3, 5}`;
- citation budget in `{0.70, 0.85, 0.95}`;
- propagation depth in `{1, 2, 3}`;
- base citation graph in `{G1, G3}`.

Report Article and JP metrics separately and stratify Article results by citation degree of the expected answers.

## LightGCN objective

Add an optional multitask objective:

`loss = BPR_articles + lambda_jp * BPR_JP + lambda_anchor * anchor_loss`.

Tune `lambda_jp in {0, 0.25, 0.5, 1.0}`. Keep `lambda_jp=0` as the exact article-only baseline.

Add a graph-aware mixed negative strategy:

- `50%` random negatives;
- `50%` semi-hard cosine negatives from ranks `20-100`;
- remove all gold items;
- remove semantic one-hop neighbors of any gold item;
- fall back to random non-gold sampling when the filtered pool is empty.

Compare it with `random`, `hard_negative_cosine_top20` and `hard_negative_cosine_top50`. Do not replace those baselines.

## Evaluation protocol

- Use the existing five train folds for every graph and model choice.
- Select graph construction, propagation budgets, training hyperparameters and stopping epoch from train-fold validation only.
- For the final replay, use the median retained epoch across folds, train on the full strict train set for that fixed number of epochs, and evaluate the strict final eval once.
- Never select a checkpoint from `eval_rich_retrievable_strict`.
- Primary metric: strict Hit@10, with MRR@10 and NDCG@10 as secondary metrics.
- Report mean and standard deviation across folds, graph coverage, and degree-stratified metrics.
- Run M3 only for final retained methods.

## Outputs

- typed graph metadata including relation edge counts and effective message mass;
- ablation tables for PPR, untrained propagation and trained LightGCN;
- training/validation curves with the retained epoch marked;
- degree-stratified Article and JP analyses;
- a final G1/G3/G5 comparison table;
- journal entry and a new presentation section after results exist.

## Acceptance criteria

- Relation channels are independently testable and backward-compatible graph variants still load unchanged.
- No final-eval checkpoint selection occurs in the new replay path.
- Every negative strategy is traceable in histories, rankings and summaries.
- The selected G5 variant improves at least one modality over its same-protocol citation baseline without regressing the other modality by more than one Hit@10 point across validation folds.
- Final claims use the untouched final eval and include fold dispersion.
