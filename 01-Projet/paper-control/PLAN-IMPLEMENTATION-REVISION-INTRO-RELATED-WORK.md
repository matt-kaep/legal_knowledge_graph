---
date: 2026-07-28
type: plan-implementation
status: termine
owner: papier
cible: superviseur
tags: [papier, cvpr, presentation, introduction, related-work]
---

# Introduction and Related Work Deck Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Réviser les slides 3 à 13 du Beamer pour développer la motivation juridique, le rôle du retrieval, une Related Work sourcée en trois slides et des contributions directement compréhensibles.

**Architecture:** L'ouverture existante est remplacée sans modifier les slides techniques postérieures. Trois slides de Related Work utilisent des tableaux `Travail / Apport / Limite`, avec références primaires lisibles. Le deck final contient exactement vingt-sept pages.

**Tech Stack:** LaTeX Beamer 16:9, thème Madrid/seahorse, `pdflatex` TeX Live 2025, Poppler pour le rendu PDF.

## Global Constraints

- Conserver le périmètre empirique du droit pénal français.
- Ne présenter aucune performance non validée comme acquise.
- Ne modifier aucun script, score ou statut expérimental.
- Ne pas affirmer que le legal case retrieval est inexistant.
- Distinguer modèles de langue, benchmarks, retrieval et graphes.
- Produire exactement vingt-sept pages sans avertissement LaTeX.

---

### Task 1: Réviser la motivation et les difficultés

**Files:**
- Modify: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

**Interfaces:**
- Consumes: slides 3 à 6 actuelles et `CONCEPTION-REVISION-INTRO-RELATED-WORK.md`.
- Produces: slide 3 sans encadré abstract, slides 4 à 6 sur le besoin juridique, la complexité et le rôle du retrieval pour les LLM.

- [x] **Step 1: Retirer l'encadré de la slide 3**
- [x] **Step 2: Réécrire la slide 4 autour des trois publics et des deux sources**
- [x] **Step 3: Réécrire la slide 5 autour des quatre difficultés juridiques**
- [x] **Step 4: Ajouter la slide 6 sur la chaîne sources, retriever, LLM et réponse ancrée**

### Task 2: Construire la Related Work en trois slides

**Files:**
- Modify: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

**Interfaces:**
- Consumes: neuf références primaires vérifiées dans la spécification.
- Produces: trois tableaux lisibles sur français/articles, jurisprudences et graphes juridiques.

- [x] **Step 1: Créer la slide français et statutory retrieval**
- [x] **Step 2: Créer la slide legal case retrieval**
- [x] **Step 3: Créer la slide graphes et réseaux de citations**
- [x] **Step 4: Afficher des références courtes et lisibles sans chiffre non nécessaire**

### Task 3: Préciser le gap et les contributions

**Files:**
- Modify: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex`

**Interfaces:**
- Consumes: gap et contributions validés dans la spécification.
- Produces: formulation limitée au droit français et trois contributions décrivant les artefacts et évaluations construits.

- [x] **Step 1: Remplacer le gap par la formulation conjointe Articles--jurisprudences**
- [x] **Step 2: Conserver l'idée principale et la question scientifique**
- [x] **Step 3: Réécrire les contributions benchmark, architectures de graphes et pipeline retrieval--LLM**
- [x] **Step 4: Supprimer les deux encadrés opaques de l'ancienne slide Contributions**

### Task 4: Compiler, inspecter et synchroniser

**Files:**
- Generate: `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.pdf`
- Modify: `01-Projet/paper-control/ETAT-PAPIER.md`
- Modify: `07-Redaction/Plan-Detaille-Papier-CVPR.md`
- Modify: `01-Projet/paper-control/CONCEPTION-REVISION-INTRO-RELATED-WORK.md`
- Modify: `01-Projet/paper-control/PLAN-IMPLEMENTATION-REVISION-INTRO-RELATED-WORK.md`

**Interfaces:**
- Consumes: Beamer révisé.
- Produces: PDF de vingt-sept pages vérifié et état rédactionnel synchronisé.

- [x] **Step 1: Compiler deux fois avec `pdflatex -halt-on-error`**
- [x] **Step 2: Vérifier vingt-sept frames, vingt-sept pages et zéro avertissement**
- [x] **Step 3: Rendre les vingt-sept pages avec `pdftoppm`**
- [x] **Step 4: Inspecter visuellement chaque page à taille lisible**
- [x] **Step 5: Documenter la révision narrative sans déclarer de nouvelle preuve**
- [x] **Step 6: Passer les statuts de conception et d'implémentation à terminé**
