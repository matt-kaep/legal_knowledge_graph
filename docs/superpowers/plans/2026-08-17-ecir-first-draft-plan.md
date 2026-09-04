# ECIR 2027 First-Draft Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a compilable first narrative draft of the ECIR 2027 paper while preserving the existing outline and keeping unsupported quantitative claims out of reader-facing prose.

**Architecture:** Create one new LNCS manuscript and one dedicated BibTeX database beside the existing outline. Draft the paper in scientific-dependency order, compile after every section, and keep the Results section explicit about unavailable validated evidence. Update the paper-control state only after the manuscript passes textual and visual verification.

**Tech Stack:** LaTeX LNCS, BibTeX with `splncs04.bst`, repository Markdown evidence notes, Poppler (`pdftoppm`, `pdftotext`, `pdfinfo`), TeX Live 2025 `pdflatex`.

**Spec:** `docs/superpowers/specs/2026-08-17-ecir-first-draft-design.md`

## Global Constraints

- Create `main-first-draft.tex`; do not modify or replace `main.tex` or `main-redline.tex`.
- Write the manuscript in English and keep Matthieu Kaeppelin and Jhony Giraldo as coordination-version authors.
- Reuse the validated mathematical notation from `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex`.
- Do not expose `experiment_id`, `E002`, `E014`, or other internal evidence identifiers in reader-facing prose or tables.
- Do not present exploratory diagnostics as main results or the already-consulted internal evaluation set as an unseen test set.
- Keep article retrieval and case-law retrieval as separate tasks with separate metrics and conclusions.
- Do not introduce a numeric result unless `paper-control` authorizes that exact interpretation.
- Compile with `/usr/local/texlive/2025/bin/universal-darwin/pdflatex`.
- Update only Task B coordination files: `ETAT-PAPIER.md` and `SYNC-PAPIER-VERS-ASSAINISSEMENT.md`.

---

### Task 1: Establish the bibliography and manuscript scaffold

**Files:**
- Create: `07-Redaction/ECIR-2027/references-first-draft.bib`
- Create: `07-Redaction/ECIR-2027/main-first-draft.tex`
- Read: `07-Redaction/ECIR-2027/main.tex`
- Read: `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex`
- Read: `02-Etat-de-l-art/Benchmarks/Louis-2022-BSARD.md`
- Read: `02-Etat-de-l-art/Benchmarks/Zheng-2021-CaseHOLD.md`
- Read: `02-Etat-de-l-art/Prediction-citations/Tang-2024-CaseLink-Inductive-Graph-Learning.md`
- Read: `02-Etat-de-l-art/NLP-juridique/Douka-2021-JuriBERT-French-Legal.md`
- Read: `02-Etat-de-l-art/gnn/LightGCN-2020.md`

**Interfaces:**
- Consumes: the section structure and notation of the existing outline and detailed plan.
- Produces: a compiling LNCS manuscript shell and stable citation keys used by every later task.

- [ ] **Step 1: Inventory primary-source metadata**

Read the complete local notes for BSARD, CaseHOLD, CaseLink, JuriBERT, and LightGCN. Extract only metadata that is explicitly recorded: authors, title, venue, year, DOI or canonical URL. Search the local state-of-the-art notes for Finding the Law, LePaRD, and CLERC; if a primary-source record is absent, record the missing reference in a source audit comment outside reader-facing prose rather than inventing metadata.

- [ ] **Step 2: Create the dedicated BibTeX database**

Create `references-first-draft.bib` with one verified entry per primary source. Use stable keys:

```bibtex
@inproceedings{louis2022bsard,
  author = {Antoine Louis and Gerasimos Spanakis},
  title = {A Statutory Article Retrieval Dataset in French},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year = {2022},
  doi = {10.18653/v1/2022.acl-long.468},
  url = {https://aclanthology.org/2022.acl-long.468/}
}
```

Repeat the complete verified metadata for `douka2021juribert`, `zheng2021casehold`, `tang2024caselink`, and `he2020lightgcn`. Add other keys only after verifying their primary records.

- [ ] **Step 3: Create the first-draft source**

Create `main-first-draft.tex` from the structural elements of `main.tex`: LNCS class, packages, title, running title, authors, the current empty `\institute{}` field, section order, bibliography style, and bibliography command. Replace outline bullet lists with empty section bodies containing only descriptive LaTeX comments that identify the paragraph map; do not copy reader-facing outline bullets into the PDF.

Required ending:

```tex
\bibliographystyle{splncs04}
\bibliography{references-first-draft}
\end{document}
```

- [ ] **Step 4: Compile the scaffold**

Run from `07-Redaction/ECIR-2027`:

```bash
/usr/local/texlive/2025/bin/universal-darwin/pdflatex \
  -interaction=nonstopmode -halt-on-error -file-line-error \
  -jobname=papier-MK-ECIR-2027-first-draft main-first-draft.tex
```

Expected: exit code 0 and `papier-MK-ECIR-2027-first-draft.pdf` exists.

- [ ] **Step 5: Commit the scaffold**

```bash
git add 07-Redaction/ECIR-2027/main-first-draft.tex \
  07-Redaction/ECIR-2027/references-first-draft.bib
git commit -m "docs: scaffold ECIR first draft"
```

---

### Task 2: Draft the Introduction and contribution contract

**Files:**
- Modify: `07-Redaction/ECIR-2027/main-first-draft.tex`
- Read: `01-Projet/paper-control/ETAT-PAPIER.md`
- Read: `01-Projet/paper-control/ETAT-ASSAINISSEMENT.md`
- Read: `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex:80`

**Interfaces:**
- Consumes: the title, scope, bibliography keys, and approved central research question.
- Produces: the motivation, scientific gap, research question, hypothesis, and contributions that constrain all later sections.

- [ ] **Step 1: Write the paragraph map as source comments**

Add six comments immediately below `\section{Introduction}`:

```tex
% P1: European and jurisdiction-specific legal-information context.
% P2: Complementary roles of statutory articles and case-law decisions.
% P3: Retrieval difficulty and grounding risk.
% P4: Precise scientific gap without an absolute novelty claim.
% P5: Research question, proposed framework, and testable hypothesis.
% P6: Contributions and empirical scope.
```

- [ ] **Step 2: Draft paragraphs 1--3**

Write connected prose establishing that legal questions are expressed in natural language while authoritative information is distributed across jurisdiction-specific statutes and decisions. Explain the normative role of statutory articles and the interpretive or applicative role of case law. Motivate retrieval as a prerequisite for grounded downstream assistance without claiming that retrieval alone solves legal reasoning.

- [ ] **Step 3: Draft paragraphs 4--6**

State the gap as underrepresentation of French article-plus-decision retrieval in reproducible evaluation, not as the non-existence of all related systems. Introduce the research question: whether relational structure can support retrieval beyond semantic similarity. State exactly two contributions: the two-task French criminal-law benchmark and the adaptable graph-based retrieval/reranking framework. Limit empirical scope to French criminal law and internal evaluation.

- [ ] **Step 4: Review paragraph contracts with the user**

For each paragraph, present its role, main claim, evidence requirement, and forbidden overclaim, followed by the English prose. Apply approved corrections before proceeding.

- [ ] **Step 5: Compile and inspect the Introduction**

Run `pdflatex` with the Task 1 command. Search the log:

```bash
rg -n "Overfull|LaTeX Error|Undefined control sequence|Fatal error" \
  papier-MK-ECIR-2027-first-draft.log
```

Expected: no matches.

- [ ] **Step 6: Commit the Introduction**

```bash
git add 07-Redaction/ECIR-2027/main-first-draft.tex
git commit -m "docs: draft ECIR introduction"
```

---

### Task 3: Draft Tasks and Benchmark

**Files:**
- Modify: `07-Redaction/ECIR-2027/main-first-draft.tex`
- Read: `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex:168`
- Read: `01-Projet/paper-control/PROTOCOLE-CONFIRMATOIRE.md`
- Read: current benchmark manifests named by `ETAT-ASSAINISSEMENT.md`

**Interfaces:**
- Consumes: the two-task contribution and validated notation.
- Produces: formal task definitions, benchmark provenance narrative, candidate-space boundaries, and a statistics table whose unavailable cells are explicitly marked pending.

- [ ] **Step 1: Draft the formal task definition**

Reuse the validated equations for `\mathcal{Q}`, `s_m`, `Y_m(q)`, and `R_m^K(q)`. Follow every displayed equation block with definitions of all variables. Define task `A` as statutory-article retrieval and task `J` as case-law-decision retrieval. State that rankings and metrics are never merged across modalities.

- [ ] **Step 2: Draft benchmark provenance and construction**

Describe the roles of legal doctrine, LEGI statutory articles, and Judilibre decisions. Explain question construction or augmentation, reference normalization, resolution against candidate corpora, and exclusion of non-retrievable references. Do not insert counts until they are traced to current versioned artifacts.

- [ ] **Step 3: Add the benchmark table**

Create a compact LNCS table with rows for questions, article candidates, case-law candidates, article relevance references, and case-law relevance references; columns distinguish training/validation material from internal evaluation. Use `\emph{pending verified export}` for any unavailable number, never a guessed value.

- [ ] **Step 4: Review paragraph contracts with the user**

Review the formal definition, provenance, construction, and scope paragraphs individually before accepting the section.

- [ ] **Step 5: Compile and commit**

Compile with the Task 1 command, require no LaTeX errors or overfull boxes, then commit:

```bash
git add 07-Redaction/ECIR-2027/main-first-draft.tex
git commit -m "docs: draft ECIR tasks and benchmark"
```

---

### Task 4: Draft the graph-based retrieval and reranking framework

**Files:**
- Modify: `07-Redaction/ECIR-2027/main-first-draft.tex`
- Read: `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex:258`
- Read: `03-Concepts/CONTEXT.md` if present

**Interfaces:**
- Consumes: task-specific candidate spaces and notation.
- Produces: the generic four-block framework, typed-graph definition, candidate scoring, and optional reranking stage used by the experiment section.

- [ ] **Step 1: Draft the framework overview**

Explain the four replaceable blocks: question/document encoding, graph exploitation or propagation, task-specific scoring, and optional reranking. Separate the generic framework from the G0--G7 experimental instantiations.

- [ ] **Step 2: Insert and define the framework equations**

Reuse `\phi_q`, `\phi_d`, `T_{\theta_m}`, `g_m`, `C_m^K`, `\rho_m`, and `S_m^{K_{\mathrm{out}}}`. Define every symbol immediately after the connected equation block. Keep `K`, `K_{\mathrm{out}}`, and `L_{\mathrm{GNN}}` distinct.

- [ ] **Step 3: Draft graph representation and candidate retrieval**

Define `G_m=(V_m,E_m,\tau_m)` and relation types. Explain citations and optional semantic neighbourhoods as observable signals, without claiming that any relation family is superior. Describe cosine scoring, PPR, and LightGCN as experimental implementations of the shared interface.

- [ ] **Step 4: Draft optional LLM reranking**

State that the reranker receives a retrieved candidate pool without hidden relevance labels and returns at most `K_{\mathrm{out}}` ordered sources. Keep answer generation downstream and outside the primary evaluation target.

- [ ] **Step 5: Review, compile, and commit**

Review each framework paragraph with the user, compile without errors or overfull boxes, then commit:

```bash
git add 07-Redaction/ECIR-2027/main-first-draft.tex
git commit -m "docs: draft ECIR retrieval framework"
```

---

### Task 5: Draft the experimental protocol and metrics

**Files:**
- Modify: `07-Redaction/ECIR-2027/main-first-draft.tex`
- Read: `01-Projet/paper-control/PROTOCOLE-CONFIRMATOIRE.md`
- Read: `05-Technique/benchmark/etape1_embedding_pur/scripts/metrics.py`
- Read: `05-Technique/benchmark/etape1_embedding_pur/scripts/graph_protocol.py`
- Read: `07-Redaction/Plan-Papier-Une-Colonne-2026-07-28.tex:375`

**Interfaces:**
- Consumes: the benchmark split, method families, and task rankings.
- Produces: reproducible model-selection rules and exact definitions of Recall, project-specific Hit, MRR, and NDCG.

- [ ] **Step 1: Draft data separation and model selection**

Describe the shared grouped five-fold validation, provenance and normalized-text grouping, validation-only hyperparameter selection, frozen replay configuration, and later internal evaluation. State that the internal evaluation has been consulted historically.

- [ ] **Step 2: Draft compared methods and implementation details**

Describe the semantic cosine control, PPR, LightGCN, and G0--G7 families at a level useful to reproduce the comparison. Move exhaustive grids to a compact table or appendix candidate. Do not expose internal experiment identifiers.

- [ ] **Step 3: Insert exact metric equations**

Copy the validated equations from the detailed plan and verify them against `metrics.py`. Explicitly state that project `Hit@K` is per-question reachable coverage,

```tex
\frac{|\widehat{R}_J^K(q_i)\cap Y_J(q_i)|}
     {\min(|Y_J(q_i)|,K)},
```

not a binary any-hit indicator. Define de-duplication, empty-GT exclusion, binary relevance, MRR capping, and NDCG normalization.

- [ ] **Step 4: Review, test, compile, and commit**

Run:

```bash
python3 -m pytest \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_metrics.py -q
```

Expected: all tests pass. Review the section paragraph by paragraph, compile without errors or overfull boxes, then commit:

```bash
git add 07-Redaction/ECIR-2027/main-first-draft.tex
git commit -m "docs: draft ECIR experimental protocol"
```

---

### Task 6: Draft Related Work from verified primary sources

**Files:**
- Modify: `07-Redaction/ECIR-2027/main-first-draft.tex`
- Modify: `07-Redaction/ECIR-2027/references-first-draft.bib`
- Read: relevant complete notes under `02-Etat-de-l-art/`

**Interfaces:**
- Consumes: the final task definition and framework positioning from Tasks 2--5.
- Produces: three evidence-backed comparison subsections and a proportionate positioning statement.

- [ ] **Step 1: Audit each comparison claim**

For every cited work, verify jurisdiction, query type, retrieved unit, graph role, supervision, and evaluation target from its primary-source note or paper. Distinguish retrieval from holding selection, citation prediction, judgment prediction, and answer generation.

- [ ] **Step 2: Draft legal information retrieval and benchmarks**

Position JuriBERT as a French legal encoder, BSARD as Belgian statutory article retrieval, and Finding the Law as hierarchy-enhanced article retrieval. Do not describe an encoder as a benchmark or a benchmark as a complete retrieval system.

- [ ] **Step 3: Draft case-law and graph-based retrieval**

Position CaseHOLD, CaseLink, LePaRD, and CLERC by their actual tasks and jurisdictions. Discuss citation and semantic graphs only when the graph directly supports retrieval; separate graph retrieval from link or judgment prediction.

- [ ] **Step 4: Draft positioning**

State the contribution as a French two-task evaluation and common graph-based interface for statutory and case-law sources. Use qualified wording such as `to our knowledge` only if the completed audit supports it; otherwise describe the observed coverage gap without an absolute novelty claim.

- [ ] **Step 5: Run BibTeX and inspect citations**

Run from `07-Redaction/ECIR-2027`:

```bash
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode \
  -halt-on-error -jobname=papier-MK-ECIR-2027-first-draft main-first-draft.tex
/usr/local/texlive/2025/bin/universal-darwin/bibtex \
  papier-MK-ECIR-2027-first-draft
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode \
  -halt-on-error -jobname=papier-MK-ECIR-2027-first-draft main-first-draft.tex
/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode \
  -halt-on-error -jobname=papier-MK-ECIR-2027-first-draft main-first-draft.tex
```

Expected: no undefined citation warning.

- [ ] **Step 6: Review and commit**

Review each Related Work paragraph with the user, then commit:

```bash
git add 07-Redaction/ECIR-2027/main-first-draft.tex \
  07-Redaction/ECIR-2027/references-first-draft.bib
git commit -m "docs: draft ECIR related work"
```

---

### Task 7: Draft limitations, future work, results status, conclusion, and abstract

**Files:**
- Modify: `07-Redaction/ECIR-2027/main-first-draft.tex`
- Read: latest `01-Projet/paper-control/ETAT-ASSAINISSEMENT.md`
- Read: latest `01-Projet/paper-control/SYNC-ASSAINISSEMENT-VERS-PAPIER.md`

**Interfaces:**
- Consumes: all accepted prose and the current scientific evidence status.
- Produces: a complete first manuscript whose claims do not exceed the available evidence.

- [ ] **Step 1: Draft limitations and future work**

Cover French-criminal-law scope, consulted internal evaluation, incomplete relevance sets, generated or augmented questions, confounded historical graph variants, LLM sensitivity, and temporal validity. Map each limitation to a concrete future-work direction.

- [ ] **Step 2: Draft the Results section status**

Describe the comparison questions, tables, and diagnostic analyses that will be reported. Where validated values are unavailable, write a visible editorial sentence stating that the quantitative table is pending final validated exports; do not use synthetic numbers or exploratory scores as substitutes.

- [ ] **Step 3: Draft the conclusion**

Restate the two-task problem and contributions. Conclude only on what the benchmark and framework define; do not claim empirical superiority while results remain pending. Limit transfer claims to future work.

- [ ] **Step 4: Rewrite the Abstract last**

Write one compact paragraph covering context, problem, benchmark, framework, evaluation design, current evidence scope, and limitations. Do not include a numerical result until authorized. Remove `Outline only` and all bullet formatting.

- [ ] **Step 5: Review, compile, and commit**

Review these sections paragraph by paragraph. Run the full BibTeX compilation sequence from Task 6, then commit:

```bash
git add 07-Redaction/ECIR-2027/main-first-draft.tex
git commit -m "docs: complete ECIR first narrative draft"
```

---

### Task 8: Verify the PDF and synchronize paper-control

**Files:**
- Modify: `01-Projet/paper-control/ETAT-PAPIER.md`
- Modify: `01-Projet/paper-control/SYNC-PAPIER-VERS-ASSAINISSEMENT.md`
- Verify: `07-Redaction/ECIR-2027/papier-MK-ECIR-2027-first-draft.pdf`

**Interfaces:**
- Consumes: the complete compiled first draft.
- Produces: a visually verified PDF and an updated Task B coordination state.

- [ ] **Step 1: Run textual safety scans**

```bash
pdftotext 07-Redaction/ECIR-2027/papier-MK-ECIR-2027-first-draft.pdf - | \
  rg -n "experiment_id|E[0-9]{3}|Outline only|Undefined citation"
```

Expected: no matches.

Scan for prohibited empirical language while results are pending:

```bash
pdftotext 07-Redaction/ECIR-2027/papier-MK-ECIR-2027-first-draft.pdf - | \
  rg -n -i "significantly outperforms|state of the art|best-performing|proves that"
```

Expected: no unsupported match.

- [ ] **Step 2: Check PDF and log metadata**

```bash
pdfinfo 07-Redaction/ECIR-2027/papier-MK-ECIR-2027-first-draft.pdf
rg -n "Overfull|LaTeX Error|Undefined control sequence|Fatal error|undefined citations" \
  07-Redaction/ECIR-2027/papier-MK-ECIR-2027-first-draft.log
```

Expected: A4 pages, successful PDF metadata, and no error or overfull match.

- [ ] **Step 3: Render and inspect every page**

```bash
mkdir -p tmp/pdfs/ecir-first-draft
pdftoppm -r 120 -png \
  07-Redaction/ECIR-2027/papier-MK-ECIR-2027-first-draft.pdf \
  tmp/pdfs/ecir-first-draft/page
```

Inspect title and authors, abstract, equations and variable definitions, tables, references, page breaks, headers, and conclusion. Correct clipping, collisions, unreadable tables, or excessive whitespace, then recompile and rerender affected pages.

- [ ] **Step 4: Update Task B coordination**

Record the new source/PDF paths, drafted sections, unresolved result and citation gaps, and absence of new scientific evidence. Tell Task A which result-table fields remain needed; do not modify Task A-owned files.

- [ ] **Step 5: Commit coordination and final manuscript state**

```bash
git add 01-Projet/paper-control/ETAT-PAPIER.md \
  01-Projet/paper-control/SYNC-PAPIER-VERS-ASSAINISSEMENT.md \
  07-Redaction/ECIR-2027/main-first-draft.tex \
  07-Redaction/ECIR-2027/references-first-draft.bib
git commit -m "docs: verify ECIR first draft"
```
