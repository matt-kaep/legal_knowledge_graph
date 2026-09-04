---
date: 2026-07-28
type: plan-implementation
status: termine
owner: papier
cible: superviseur
tags: [papier, cvpr, presentation, beamer, questions-reviewer]
---

# Reviewer-Question Beamer Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réorganiser le support Beamer du plan CVPR en vingt-quatre slides guidées par les questions du reviewer, avec la section du papier explicitement indiquée sur chaque slide.

**Architecture:** Le fichier Beamer existant reste la source unique du support courant. Les contenus scientifiques validés sont conservés, l'ouverture est décomposée en quatre fonctions argumentatives et les titres techniques sont reformulés en questions. Une macro visuelle commune porte le nom de la section du papier sans concurrencer le titre principal.

**Tech Stack:** LaTeX Beamer 16:9, thème Madrid/seahorse, `pdflatex` TeX Live 2025, Poppler `pdfinfo` et `pdftoppm`.

## Global Constraints

- Ne modifier aucun script, score, artefact ou statut expérimental.
- Ne présenter aucun résultat exploratoire comme confirmatoire.
- N'afficher aucun nouveau chiffre sans `experiment_id` validé par la tâche A.
- Conserver la séparation entre confirmation interne et future lockbox.
- Conserver le style des présentations hebdomadaires existantes.
- Le support final contient exactement vingt-quatre pages.

---

### Task 1: Installer la grammaire questions-sections

**Files:**
- Modify: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

**Interfaces:**
- Consumes: préambule Beamer existant et décision décrite dans `CONCEPTION-DECK-QUESTIONS-REVIEWER.md`.
- Produces: macro `\papersection{...}` et carte d'ouverture reliant les questions aux sections.

- [x] **Step 1: Ajouter la macro de section**

Ajouter une macro `\papersection{}` qui affiche un libellé discret en couleur d'accent immédiatement sous le titre de chaque slide de contenu.

- [x] **Step 2: Remplacer le plan de séance**

La slide 2 explique la lecture `question du reviewer -> réponse -> section du papier`. La slide 3 montre les huit sections du manuscrit : Introduction, Related Work, Task and Benchmark, Method, Experiments, Results, Discussion and Limitations, Conclusion.

- [x] **Step 3: Vérifier le nombre de frames intermédiaire**

Run: `rg -c '^\\begin\{frame\}' 01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

Expected: un nombre temporaire cohérent avec les slides déjà remplacées, sans frame dupliquée accidentellement.

### Task 2: Reconstruire l'ouverture argumentative

**Files:**
- Modify: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

**Interfaces:**
- Consumes: contenu actuel des slides Introduction, question centrale, contributions et Related Work.
- Produces: slides 4 à 10 dans l'ordre motivation, difficulté, familles existantes, manque, intuition, question et contributions.

- [x] **Step 1: Séparer motivation et difficulté**

La motivation expose la double recherche Articles/JP. La difficulté expose la volumétrie, l'hétérogénéité, les deux espaces candidats et la faiblesse des liens directement exploitables.

- [x] **Step 2: Séparer état de l'art et manque**

Une slide décrit les trois familles pertinentes ; la suivante formule uniquement le manque que la Related Work devra soutenir, sans affirmer encore une nouveauté non vérifiée.

- [x] **Step 3: Isoler l'intuition principale**

La slide d'intuition présente le graphe typé comme articulation des citations et proximités sémantiques, puis annonce les deux étages retrieval et LLM reranking.

- [x] **Step 4: Conserver question, portée et contributions**

Reformuler les titres en questions et préserver les réserves sur la portée pénale et la contribution de performance conditionnelle.

### Task 3: Reformer les blocs techniques et expérimentaux

**Files:**
- Modify: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

**Interfaces:**
- Consumes: slides existantes sur la tâche, le benchmark, $G_0$--$G_7$, la formalisation, les méthodes, le protocole, les résultats et les limites.
- Produces: slides 11 à 24, chacune titrée par une question et étiquetée par section du papier.

- [x] **Step 1: Renommer les slides Task and Benchmark**

Utiliser `Quelle tâche évaluons-nous ?` puis `Comment le benchmark français est-il construit ?`.

- [x] **Step 2: Ordonner la Method du général au détaillé**

Présenter successivement l'inventaire des graphes, la formalisation du graphe typé, le premier étage de retrieval, la capacité contrôlée du LLM et le pipeline end-to-end.

- [x] **Step 3: Reformuler le bloc Experiments and Results**

Présenter le protocole, les métriques/baselines/ablations, le programme des tableaux et le programme des figures comme réponses aux affirmations à tester.

- [x] **Step 4: Ajouter la slide Conclusion puis Abstract**

Créer une slide qui indique le rôle futur de ces deux sections sans rédiger leurs affirmations finales.

- [x] **Step 5: Conserver la slide des décisions**

La slide 24 reste la sortie opérationnelle de la discussion avec le superviseur.

### Task 4: Compiler et vérifier le support

**Files:**
- Modify if needed: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`
- Generate: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.pdf`

**Interfaces:**
- Consumes: source Beamer restructurée.
- Produces: PDF 16:9 de vingt-quatre pages sans défaut de composition.

- [x] **Step 1: Compiler deux fois**

Run: `/usr/local/texlive/2025/bin/universal-darwin/pdflatex -interaction=nonstopmode -halt-on-error -output-directory=01-Projet/presentations 01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

Expected: compilation réussie lors des deux passes.

- [x] **Step 2: Contrôler le log et la pagination**

Run: `rg 'Overfull|Underfull|LaTeX Warning|Package .* Warning' 01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.log`

Expected: aucune correspondance.

Run: `pdfinfo 01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.pdf | rg '^Pages:'`

Expected: `Pages: 24`.

- [x] **Step 3: Rendre toutes les pages**

Run: `pdftoppm -png -r 150 01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.pdf tmp/pdfs/plan-latex-reviewer/slide`

Expected: vingt-quatre images PNG.

- [x] **Step 4: Inspecter chaque page**

Vérifier les vingt-quatre pages à taille lisible : titres, libellés de section, tableaux, figures, encadrés, pagination, absence de coupure et lisibilité des légendes.

### Task 5: Synchroniser la source de vérité papier

**Files:**
- Modify: `01-Projet/paper-control/ETAT-PAPIER.md`
- Modify: `07-Redaction/Plan-Detaille-Papier-CVPR.md`
- Modify: `01-Projet/paper-control/PLAN-IMPLEMENTATION-DECK-QUESTIONS-REVIEWER.md`

**Interfaces:**
- Consumes: deck final vérifié.
- Produces: état rédactionnel et plan détaillé alignés sur l'architecture questions-sections.

- [x] **Step 1: Documenter la décision**

Indiquer que l'option C est validée et que le support courant suit les questions du reviewer en vingt-quatre slides.

- [x] **Step 2: Conserver les limites scientifiques**

Mentionner que la réorganisation n'ajoute aucune preuve et ne change aucun statut expérimental.

- [x] **Step 3: Fermer le plan d'implémentation**

Passer le frontmatter de ce document de `en-cours` à `termine` après réussite de toutes les vérifications.
