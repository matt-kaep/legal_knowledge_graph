---
date: 2026-08-11
type: specification
status: approved-design
tags: [papier, mathematiques, notation, latex]
---

# Mathematical notation and variable definitions

## Objective

Make every displayed mathematical block independently understandable by defining every set, index, scalar, vector, function, operator, and output used in that block.

## Scope

Apply the convention to the active writing artifacts:

- `07-Redaction/ECIR-2027/main.tex`;
- `07-Redaction/ECIR-2027/main-redline.tex`;
- `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex`.

Update the paper state and outgoing synchronization channel after the LaTeX changes. Do not rename graph identifiers G0--G7 or change any experiment, metric, scientific status, or quantitative claim.

## Presentation rule

Every displayed equation or logically connected equation block is followed immediately by a compact `where:` paragraph or list. That block defines every symbol appearing in the equation, even when a short definition is repeated from an earlier page.

Definitions must include, where applicable:

- semantic meaning;
- set membership or function domain and codomain;
- vector or matrix dimension;
- index range;
- distinction between task-specific and shared quantities.

No symbol may first appear only through an unexplained acronym or implementation name.

## Canonical question notation

Define the benchmark questions as

\[
\mathcal Q=\{q_i\}_{i=1}^{N},\qquad N=|\mathcal Q|,
\]

where \(\mathcal Q\) is the set of legal questions, \(N\) is the number of questions, \(i\in\{1,\ldots,N\}\) is a question index, and \(q_i\) is the \(i\)-th question. Subsequent model equations may use \(q\in\mathcal Q\) as a generic question.

## Canonical symbols

### Retrieval task

- \(m\in\{A,J\}\): retrieval task, with \(A\) for statutory articles and \(J\) for case-law decisions.
- \(\mathcal C_m\): candidate-source set for task \(m\).
- \(d\in\mathcal C_m\): generic candidate source.
- \(Y_m(q)\subseteq\mathcal C_m\): sources relevant to question \(q\) for task \(m\).
- \(s_m(q,d)\in\mathbb R\): retrieval score assigned to candidate \(d\).
- \(K\in\mathbb N_{>0}\): candidate-pool depth.
- \(\operatorname{Seq}_r(\mathcal C_m)\): ordered sequences of \(r\) distinct candidates from \(\mathcal C_m\).
- \(\operatorname{TopK}\): operator returning the \(K\) highest-scoring candidates in descending score order.
- \(R_m^K(q)\in\operatorname{Seq}_K(\mathcal C_m)\): ordered top-\(K\) ranking for task \(m\).

### Encoders and graph operator

- \(p\in\mathbb N_{>0}\): embedding dimension.
- \(\phi_q:\mathcal Q\rightarrow\mathbb R^p\): question encoder.
- \(\phi_d:\mathcal C_m\rightarrow\mathbb R^p\): candidate-source encoder.
- \(\mathbf z_q=\phi_q(q)\in\mathbb R^p\): representation of question \(q\).
- \(\mathbf h_d\in\mathbb R^p\): graph-aware representation of candidate \(d\).
- \(T_{\theta_m}\): task-specific graph operator, parameterized by \(\theta_m\), that maps \(G_m\) and the candidate-source embeddings to graph-aware embeddings of the same candidates.

### Graph representation

- \(G_m=(V_m,E_m,\tau_m)\): graph available to task \(m\).
- \(V_m\): typed node set containing the available statutory articles and judicial decisions.
- \(E_m\subseteq V_m\times V_m\): edge set.
- \(\mathcal R_m\): set of relation types available to task \(m\).
- \(\tau_m:E_m\rightarrow\mathcal R_m\): edge-type function.

### Scoring and reranking

- \(g_m:\mathbb R^p\times\mathbb R^p\rightarrow\mathbb R\): task-specific candidate scorer.
- \(C_m^K(q)\in\operatorname{Seq}_K(\mathcal C_m)\): ordered candidate pool of depth \(K\).
- \(K_{\mathrm{out}}\in\{1,\ldots,K\}\): maximum reranked output depth.
- \(\rho_m:\mathcal Q\times\operatorname{Seq}_K(\mathcal C_m)\rightarrow\operatorname{Seq}_{K_{\mathrm{out}}}(\mathcal C_m)\): optional reranking function for task \(m\).
- \(S_m^{K_{\mathrm{out}}}(q)\in\operatorname{Seq}_{K_{\mathrm{out}}}(\mathcal C_m)\): ordered reranked subset.

### Plan-specific graph and learning notation

- \(A_{G7}\), \(A_{G1}^{\mathrm{cit}}\), and \(A_{G4,b}^{\mathrm{sem}}\): adjacency matrices for G7, citation graph G1, and semantic block \(b\) of G4.
- \(\mathcal B\subseteq\{AA,JJ,AJ\}\): selected semantic-relation blocks.
- \(\lambda_{\mathrm{cit}}\) and \(\lambda_b\): non-negative relation-family weights.
- \(\widetilde A_{G7}\): normalized adjacency matrix used for propagation.
- \(H^{(\ell)}\): node-embedding matrix after propagation layer \(\ell\).
- \(L_{\mathrm{GNN}}\): number of graph-propagation layers.
- \(H^*\): mean representation across layers \(0\) through \(L_{\mathrm{GNN}}\).
- \(\mathbf z_q\) and \(\mathbf h_a^*\): question embedding and final graph-aware embedding of article \(a\).
- \(a^+\) and \(a^-\): relevant and sampled non-relevant articles for a training triple.
- \(\sigma\): logistic sigmoid.
- \(\tau_{\mathrm{BPR}}>0\): BPR temperature.
- \(\mathcal L_{\mathrm{BPR}}\): Bayesian Personalized Ranking loss.

## Collision prevention

- Reserve \(K\) for candidate-pool depth.
- Replace the reranked-list length \(L\) with \(K_{\mathrm{out}}\).
- Replace the LightGCN layer count \(L\) with \(L_{\mathrm{GNN}}\).
- Keep \(\tau_m\) for graph edge typing and write \(\tau_{\mathrm{BPR}}\) for the loss temperature.
- Replace layer embeddings \(E^{(\ell)}\) and \(E^*\) with \(H^{(\ell)}\) and \(H^*\) so they cannot be confused with the edge set \(E_m\).
- Use \(d\) only for a generic candidate source; use \(a\) and \(j\) when an equation is specific to an article or a judicial decision.

## Redline behavior

The clean source shows the final equations and definitions normally. The redline preserves removed symbols and wording in black strikethrough and shows new symbols, equations, and `where:` definitions in red. Long definition blocks may use separate deleted and added lines to avoid overfull boxes.

## Verification

Compile each affected LaTeX source twice with the project TeX Live binary. Require:

- successful compilation;
- no fatal, undefined-control-sequence, emergency-stop, or overfull-box warning;
- stable A4 pagination;
- text extraction containing the expected definitions;
- visual inspection of every affected page in the clean, redline, and one-column PDFs;
- valid CSV and coordination files when their wording is synchronized.

## Non-goals

- No selection of graph, encoder, hyperparameter, or model.
- No new empirical result or confirmatory claim.
- No change to G0--G7 identifiers or experiment IDs.
- No global notation appendix unless later requested.
