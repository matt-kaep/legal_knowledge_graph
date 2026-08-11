---
date: 2026-08-11
type: implementation-plan
status: ready
tags: [papier, mathematiques, notation, latex]
---

# Mathematical Notation Definitions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every displayed mathematical block in the active ECIR outline and one-column plan independently understandable by defining every variable, set, index, function, operator, and dimension immediately after the block.

**Architecture:** Establish the final notation in the clean ECIR source first, apply the same convention to every equation in the detailed one-column plan, and then mirror the clean ECIR changes in the manually maintained redline. Keep scientific content and experiment identifiers unchanged; only notation, explanatory definitions, and collision-free symbol names change.

**Tech Stack:** LaTeX, Springer LNCS, `latexdiff`-style `\DIFadd`/`\DIFdel` markup, TeX Live 2025 `pdflatex`, Poppler `pdftotext`/`pdftoppm`/`pdfinfo`, Ruby CSV validation.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-11-mathematical-notation-definitions-design.md` exactly.
- Define `\mathcal Q=\{q_i\}_{i=1}^{N}` and `N=|\mathcal Q|`, then use generic `q\in\mathcal Q` in model equations.
- Every displayed equation or connected equation block must be followed by a self-contained `\textbf{where:}` list defining every symbol in that block.
- Use `K` only for candidate-pool depth, `K_{\mathrm{out}}` for reranked depth, and `L_{\mathrm{GNN}}` for propagation depth.
- Use `\tau_m:E_m\to\mathcal R_m` for edge typing and `\tau_{\mathrm{BPR}}` for BPR temperature.
- Use `H^{(\ell)}` and `H^*` for layer embeddings; reserve `E_m` for graph edges.
- Do not rename G0--G7, alter an `experiment_id`, add a result, or change a scientific status.
- Preserve unrelated dirty and untracked work. Do not commit or push implementation files unless the user explicitly requests it.
- Compile with `/usr/local/texlive/2025/bin/universal-darwin/pdflatex`.

---

### Task 1: Make the clean ECIR mathematical formulation self-contained

**Files:**
- Modify: `07-Redaction/ECIR-2027/main.tex:84-142`

**Interfaces:**
- Consumes: canonical symbols and collision rules from the approved design specification.
- Produces: the final clean equations and `where:` definitions that the redline must mirror exactly.

- [ ] **Step 1: Capture the pre-edit notation gaps**

Run:

```bash
rg -n '\\mathcal\{Q\}|R_m\^K|\\phi_q|T_\{\\theta_m\}|S_m\^L|G_m=\\left|G_m=\(' 07-Redaction/ECIR-2027/main.tex
```

Expected: `S_m^L` is still present; `N`, `K_{\mathrm{out}}`, `\mathcal R_m`, and explicit function domains are absent.

- [ ] **Step 2: Replace the formal task definition with indexed questions and complete definitions**

Insert the following logical structure in `Formal Task Definition`:

```latex
Let the benchmark contain
\[
  \mathcal Q=\{q_i\}_{i=1}^{N},\qquad N=|\mathcal Q|.
\]
\noindent\textbf{where:}
\begin{itemize}
  \item \(\mathcal Q\) is the set of natural-language legal questions;
  \item \(N\in\mathbb N_{>0}\) is the number of questions;
  \item \(i\in\{1,\ldots,N\}\) is a question index; and
  \item \(q_i\in\mathcal Q\) is the \(i\)-th question.
\end{itemize}
```

Keep `m\in\{A,J\}` and `Y_m(q)` in the surrounding prose, then follow the score signature with definitions of `s_m`, `m`, `\mathcal C_m`, and `\mathbb R`. Follow the ranking equation with definitions of `R_m^K(q)`, generic `q`, `K`, `\operatorname{TopK}`, generic candidate `d`, and `s_m(q,d)`.

- [ ] **Step 3: Type the encoder and graph-operator equation completely**

Keep the equation

```latex
\[
  \mathbf{z}_q=\phi_q(q),\qquad
  \{\mathbf{h}_d:d\in\mathcal{C}_m\}
  =T_{\theta_m}\!\left(G_m,\{\phi_d(d):d\in\mathcal{C}_m\}\right),
\]
```

and define immediately below it:

- `q\in\mathcal Q` as the generic legal question;
- `p\in\mathbb N_{>0}` as embedding dimension;
- `\phi_q:\mathcal Q\to\mathbb R^p` as question encoder;
- `\mathbf z_q\in\mathbb R^p` as question representation;
- `\phi_d:\mathcal C_m\to\mathbb R^p` as candidate-source encoder;
- `\mathbf h_d\in\mathbb R^p` as graph-aware representation of `d`;
- `G_m` as the task-specific graph;
- `T_{\theta_m}` as the task-specific graph operator; and
- `\theta_m` as its parameters.

- [ ] **Step 4: Remove the reranking-depth collision**

Replace `S_m^L(q)` with:

```latex
S_m^{K_{\mathrm{out}}}(q)
=\rho_m\!\left(q,C_m^K(q)ight).
```

The connected `where:` list must define `C_m^K(q)`, `K`, `g_m:\mathbb R^p\times\mathbb R^p\to\mathbb R`, `K_{\mathrm{out}}\in\{1,\ldots,K\}`, `\rho_m`, and `S_m^{K_{\mathrm{out}}}(q)` as an ordered reranked sequence.

- [ ] **Step 5: Turn the graph tuple into a defined mathematical block**

Use:

```latex
\[
  G_m=(V_m,E_m,\tau_m),\qquad
  \tau_m:E_m\rightarrow\mathcal R_m.
\]
```

Define `G_m`, `V_m`, `E_m\subseteq V_m\times V_m`, `\mathcal R_m`, and `\tau_m`. State explicitly that nodes are statutory articles or judicial decisions available to task `m`.

- [ ] **Step 6: Compile the clean ECIR PDF twice**

Run from `07-Redaction/ECIR-2027/`:

```bash
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode -halt-on-error -jobname=papier-MK-ECIR-2027 main.tex
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode -halt-on-error -jobname=papier-MK-ECIR-2027 main.tex
```

Expected: exit 0 twice. Do not accept any `Overfull`, `Undefined control sequence`, `Emergency stop`, or fatal error.

- [ ] **Step 7: Verify symbol coverage in the clean PDF**

Run:

```bash
pdftotext papier-MK-ECIR-2027.pdf - | rg 'number of questions|question encoder|embedding dimension|candidate-pool depth|reranked output|edge-type function'
```

Expected: all six descriptions are present.

---

### Task 2: Apply the convention to every equation in the detailed one-column plan

**Files:**
- Modify: `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex:90-360`

**Interfaces:**
- Consumes: the clean ECIR definitions from Task 1 and the plan-specific symbols from the design specification.
- Produces: a self-contained French planning document covering task equations, graph construction, propagation, scoring, BPR, and LLM reranking.

- [ ] **Step 1: Inventory every plan equation before editing**

Run:

```bash
rg -n '\\begin\{equation\}|\\begin\{align\}' 07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex
```

Expected: equation blocks for the introductory rankings, article retrieval, case-law retrieval, framework, G7, propagation, article scoring, BPR, and final LLM scoring.

- [ ] **Step 2: Define the introductory and task-specific ranking variables**

Before the introductory pair, add the same benchmark definition as in the ECIR outline: `\mathcal Q=\{q_i\}_{i=1}^{N}` and `N=|\mathcal Q|`, followed by definitions of `\mathcal Q`, `N`, index `i`, and question `q_i`. After the pair `q\mapsto R_A^K(q)` and `q\mapsto R_J^K(q)`, define generic question `q`, task rankings `R_A^K` and `R_J^K`, candidate sets `\mathcal C_A` and `\mathcal C_J`, and positive integer `K`.

After the article equation, define `a\in\mathcal C_A`, score `s_A(q,a)\in\mathbb R`, `\operatorname{TopK}`, and the ordered result `R_A^K(q)`. After the decision equation, repeat the complete definitions with `j\in\mathcal C_J` and `s_J(q,j)`.

- [ ] **Step 3: Define the three-stage framework equations**

Retain the align block for `C_m^K`, reranking, and answer generation, but replace the hard-coded superscript `10` with `K_{\mathrm{out}}`:

```latex
C_m^K(q) &= \operatorname{Retriever}_m(q,\mathcal C_m,G_m),\\
S_m^{K_{\mathrm{out}}}(q)
  &= \operatorname{LLMRank}_m(q,C_m^K(q)),\\
y &= \operatorname{LLMAnswer}\!\left(q,
S_A^{K_{\mathrm{out}}}(q),S_J^{K_{\mathrm{out}}}(q)\right).
```

The following `where:` list defines `m`, `q`, `\mathcal C_m`, `G_m`, `K`, `C_m^K`, `K_{\mathrm{out}}`, `S_m^{K_{\mathrm{out}}}`, and grounded answer `y`. It also defines `\operatorname{Retriever}_m` as the task-specific candidate retriever, `\operatorname{LLMRank}_m` as the task-specific reranker, and `\operatorname{LLMAnswer}` as the downstream grounded-answer function. State `1\le K_{\mathrm{out}}\le K`.

- [ ] **Step 4: Define every G7 construction symbol**

After the G7 adjacency equation, define `A_{G7}`, `A_{G1}^{\mathrm{cit}}`, and `A_{G4,b}^{\mathrm{sem}}` as matrices in `\mathbb R^{|V|\times|V|}` over the shared node set `V`; define `\mathcal B`, block index `b`, the meanings of `AA`, `JJ`, and `AJ`, and non-negative weights `\lambda_{\mathrm{cit}}` and `\lambda_b`.

- [ ] **Step 5: Rename and define the graph-propagation representations**

Replace:

```latex
E^{(\ell+1)}=\widetilde{A}_{G7}E^{(\ell)},
\qquad
E^*=\frac{1}{L+1}\sum_{\ell=0}^{L}E^{(\ell)}.
```

with:

```latex
H^{(\ell+1)}=\widetilde{A}_{G7}H^{(\ell)},
\qquad
H^*=\frac{1}{L_{\mathrm{GNN}}+1}
\sum_{\ell=0}^{L_{\mathrm{GNN}}}H^{(\ell)}.
```

Define normalized adjacency `\widetilde A_{G7}\in\mathbb R^{|V|\times|V|}`, layer index `\ell\in\{0,\ldots,L_{\mathrm{GNN}}\}`, layer count `L_{\mathrm{GNN}}\in\mathbb N_{>0}`, node-embedding matrices `H^{(\ell)}\in\mathbb R^{|V|\times p}`, shared node set `V`, embedding dimension `p`, and averaged representation `H^*\in\mathbb R^{|V|\times p}`.

- [ ] **Step 6: Normalize and define article scoring**

Replace `s_A(q,a)=\cos(e_q,E_a^*)` with:

```latex
s_A(q,a)=\cos\!\left(\mathbf z_q,\mathbf h_a^*\right).
```

Define `q`, article `a`, question embedding `\mathbf z_q\in\mathbb R^p`, final article representation `\mathbf h_a^*\in\mathbb R^p`, cosine similarity, dimension `p`, and scalar score `s_A(q,a)`.

- [ ] **Step 7: Disambiguate and define the BPR loss**

Replace the denominator `\tau` with `\tau_{\mathrm{BPR}}` and define `\mathcal L_{\mathrm{BPR}}`, generic question `q\in\mathcal Q`, sigmoid `\sigma`, positive article `a^+`, sampled non-relevant article `a^-`, score function `s`, and positive temperature `\tau_{\mathrm{BPR}}>0`.

- [ ] **Step 8: Define the final LLM scoring equation**

After `s_{\mathrm{final}}(q,d)=\operatorname{LLMRank}(q,d\mid C_m^K(q))`, define final score, generic source `d`, question `q`, task `m`, candidate pool `C_m^K(q)`, and the conditional bar as indicating that the score is computed from the provided candidate pool without hidden relevance labels.

- [ ] **Step 9: Compile the one-column plan twice**

Run from `07-Redaction/`:

```bash
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode -halt-on-error -jobname=Plan-Papier-Une-Colonne-2026-07-28 Plan-Papier-Une-Colonne-2026-07-28.tex
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode -halt-on-error -jobname=Plan-Papier-Une-Colonne-2026-07-28 Plan-Papier-Une-Colonne-2026-07-28.tex
```

Expected: exit 0 twice and no overfull or fatal warning.

---

### Task 3: Synchronize the LaTeX redline with the final clean notation

**Files:**
- Modify: `07-Redaction/ECIR-2027/main-redline.tex:150-245`

**Interfaces:**
- Consumes: exact final ECIR equations and definition text from Task 1.
- Produces: a reviewer-facing redline in which obsolete notation is struck in black and every new definition appears in red.

- [ ] **Step 1: Mirror every clean mathematical block**

For each clean block added or changed in Task 1, reproduce the same final equation and `where:` definitions inside the established `\DIFaddbegin ... \DIFaddend` structure. Keep list environments outside `\DIFadd{...}` when wrapping the entire environment would break compilation; mark the textual content of each new item with `\DIFadd`.

- [ ] **Step 2: Mark every replaced symbol explicitly**

Show these replacements as redline changes:

- `S_m^L` deleted and `S_m^{K_{\mathrm{out}}}` added;
- the previously inline `G_m=(V_m,E_m,\tau_m)` deleted and the complete displayed typed-graph block from Task 1 added;
- any vague `candidate documents` wording deleted and explicit `candidate sources` wording added.

- [ ] **Step 3: Prevent redline overfull boxes**

Where a deleted definition and its replacement do not fit on one line, render them as separate deleted and added lines, following the existing conclusion pattern:

```latex
\DIFdel{Old complete statement.}\\
\DIFadd{New complete statement.}
```

- [ ] **Step 4: Compile the redline twice**

Run from `07-Redaction/ECIR-2027/`:

```bash
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode -halt-on-error -jobname=papier-MK-ECIR-2027-redline main-redline.tex
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode -halt-on-error -jobname=papier-MK-ECIR-2027-redline main-redline.tex
```

Expected: exit 0 twice and no overfull or fatal warning.

- [ ] **Step 5: Verify the tracked notation text**

Run:

```bash
pdftotext papier-MK-ECIR-2027-redline.pdf - | rg 'number of questions|embedding dimension|reranked output depth|edge-type function'
```

Expected: all four definitions are extractable from the redline PDF.

---

### Task 4: Synchronize paper control and perform final artifact verification

**Files:**
- Modify: `01-Projet/paper-control/ETAT-PAPIER.md`
- Modify: `01-Projet/paper-control/SYNC-PAPIER-VERS-ASSAINISSEMENT.md`
- Verify: `07-Redaction/ECIR-2027/papier-MK-ECIR-2027.pdf`
- Verify: `07-Redaction/ECIR-2027/papier-MK-ECIR-2027-redline.pdf`
- Verify: `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.pdf`

**Interfaces:**
- Consumes: the final sources and PDFs from Tasks 1--3.
- Produces: current paper state, outgoing synchronization note, and verified PDF artifacts.

- [ ] **Step 1: Record the notation convention in paper state**

Add a terminology/methodology bullet stating that every mathematical block now defines every symbol immediately below it; record `N`, `K`, `K_{\mathrm{out}}`, `L_{\mathrm{GNN}}`, `\tau_m`, and `\tau_{\mathrm{BPR}}` as the collision-free convention. Update `Dernière mise à jour` to 2026-08-11 and state explicitly that no experiment or scientific status changed.

- [ ] **Step 2: Add the outgoing synchronization entry**

Add a 2026-08-11 entry explaining that the change is purely expository, listing the three LaTeX/PDF artifacts, asking task A to use the same symbols in future mathematical exports, and stating that G0--G7 and all `experiment_id` values remain unchanged.

- [ ] **Step 3: Run structural checks**

Run:

```bash
git diff --check -- \
  07-Redaction/ECIR-2027/main.tex \
  07-Redaction/ECIR-2027/main-redline.tex \
  07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex \
  01-Projet/paper-control/ETAT-PAPIER.md \
  01-Projet/paper-control/SYNC-PAPIER-VERS-ASSAINISSEMENT.md
```

Expected: exit 0 and no output.

Run:

```bash
rg -n '(^!|Fatal error|Undefined control sequence|Emergency stop|Overfull)' \
  07-Redaction/ECIR-2027/papier-MK-ECIR-2027.log \
  07-Redaction/ECIR-2027/papier-MK-ECIR-2027-redline.log \
  07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.log
```

Expected: exit 1 and no output.

- [ ] **Step 4: Verify page format and expected text**

Run `pdfinfo` on all three PDFs and require A4 page size. Run `pdftotext` and verify the expected variable descriptions in each affected document.

- [ ] **Step 5: Render and inspect every affected page**

Create `tmp/pdfs/mathematical-definitions/`, render all pages containing changed equations with `pdftoppm -png`, and inspect each PNG for clipped definitions, broken equations, overlapping redline text, and unreadable line wrapping. Delete the temporary directory after inspection.

- [ ] **Step 6: Report the completed scope**

Report the three final PDFs, the two LaTeX sources, the plan source, the notation collisions resolved, compilation/page evidence, and the fact that no scientific result or experiment identifier changed.
