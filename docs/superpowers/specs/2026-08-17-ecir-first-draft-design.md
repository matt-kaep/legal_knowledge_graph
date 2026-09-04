# ECIR 2027 First-Draft Design

Date: 2026-08-17

## Objective

Produce a first complete narrative draft of the ECIR 2027 paper from the validated one-column detailed plan and the current LNCS outline. The draft must be readable from beginning to end while remaining explicit about evidence that is not yet publishable.

## Output artifacts

- New source: `07-Redaction/ECIR-2027/main-first-draft.tex`.
- New compiled PDF: `07-Redaction/ECIR-2027/papier-MK-ECIR-2027-first-draft.pdf`.
- Existing `main.tex`, `main-redline.tex`, and their PDFs remain unchanged reference artifacts.

## Manuscript scope

The first draft will contain narrative prose for:

1. Abstract, marked provisional until the results are fixed.
2. Introduction.
3. Related Work.
4. Tasks and Benchmark.
5. Graph-Based Legal Retrieval and Reranking Framework.
6. Experiments and Results.
7. Limitations and Future Work.
8. Conclusion, written without an unsupported superiority claim.

The formal task definitions, graph notation, reranking notation, and evaluation metrics will reuse the mathematical conventions already validated in the detailed plan.

## Paragraph-level writing contract

Each paragraph must have one primary function in the argument. Before accepting a paragraph as drafted, it must be possible to identify:

- the question it answers for the reader;
- its central claim;
- the evidence or citation supporting that claim;
- the transition it creates toward the next paragraph;
- any limitation that constrains its interpretation.

Paragraphs should not mix motivation, methodological definition, empirical result, and limitation unless the connection is necessary to understand the claim.

## Evidence and citation policy

- Bibliographic claims use verified primary sources whenever available locally.
- A citation placeholder may remain only when the required source is identified precisely but has not yet been verified.
- No absolute novelty claim is allowed without a documented literature audit.
- Internal identifiers such as `experiment_id`, `E002`, or `E014` never appear in reader-facing prose or tables.
- Internal registries and `paper-control` remain the provenance layer used to decide whether a result is eligible for the manuscript.

## Quantitative-results policy

- Results that are not authorized by the scientific coordination state remain explicit placeholders.
- Exploratory diagnostics may be discussed only in the limitations or diagnostic-analysis framing explicitly authorized by Task A.
- The internal evaluation set is described as already consulted and is never presented as an unseen final test set.
- Articles and case-law results remain separate; no combined score is introduced.
- The draft must not name a globally superior graph or method unless the current evidence supports that exact claim.

## Drafting sequence

The manuscript will be drafted in dependency order rather than final reading order:

1. Introduction and contributions.
2. Tasks and Benchmark.
3. Framework and mathematical definitions.
4. Experimental protocol and metrics.
5. Related Work, once the exact positioning claims are stable.
6. Limitations and Future Work.
7. Results placeholders and permitted diagnostics.
8. Conclusion.
9. Abstract, rewritten last from the completed draft.

This sequence prevents the Abstract and Related Work from promising contributions that the formal sections do not actually support.

## Review workflow

The user and assistant will review the draft paragraph by paragraph. For each paragraph, the assistant will first provide its intended role, content, required evidence, and forbidden overclaims, then propose the English prose. Accepted prose will be kept in the first-draft source.

## Validation

Before delivery:

- compile the source twice with the repository's pinned `pdflatex` binary;
- reject LaTeX errors, undefined commands, and overfull boxes;
- render representative and changed PDF pages with Poppler;
- inspect title, authors, equations, tables, section transitions, and page boundaries;
- scan the reader-facing PDF for internal experiment identifiers and unsupported result placeholders presented as facts;
- update `ETAT-PAPIER.md` and `SYNC-PAPIER-VERS-ASSAINISSEMENT.md` when the first draft reaches a meaningful checkpoint.

## Out of scope

- Replacing the existing clean outline or redline.
- Publishing exploratory scores as main results.
- Inventing missing benchmark statistics or citations.
- Renaming technical graph artifacts G0--G7 in the scientific pipeline.
- Submitting, pushing, or publishing the manuscript externally.
