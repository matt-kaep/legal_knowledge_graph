---
tags: [article, benchmark, nlp-juridique, multilingue, francais]
categorie: "Benchmarks"
titre_complet: "LEXTREME: A Multi-Lingual and Multi-Task Benchmark for the Legal Domain"
auteurs: "Niklaus, Matoshi, Rani, Galassi, Stürmer, Chalkidis"
annee: 2023
type: "Article de conférence"
venue: "EMNLP Findings 2023"
url: "https://arxiv.org/abs/2301.13126"
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# LEXTREME — Niklaus et al. 2023

> [!info] Metadonnees
> **Auteurs** : Niklaus, Matoshi, Rani, Galassi, Stürmer, Chalkidis
> **Annee** : 2023 | **Venue** : EMNLP Findings 2023
> **URL** : [Lien](https://arxiv.org/abs/2301.13126)

## Resume

Premier benchmark NLP juridique **multilingue** — couvre **24 langues** via **11 datasets**. Comble le vide de LexGLUE (anglais) et LegalBench (anglais). Inclut le français. Meilleur modèle (XLM-R large) : score agrégé 61.3 → marge de progression.

## Langues et datasets

- **24 langues européennes** (toutes les langues officielles EU + quelques extensions)
- **11 datasets** issus de sources juridiques multi-juridictions
- Méthodologie : 2 scores agrégés (par dataset ET par langue) pour équité

## Les datasets incluant le francais — détail

Sur les **11 datasets / 18 configurations** de LEXTREME, **seulement 2 contiennent du français** :

### 1. MultiEURLEX

- **Tâche** : classification multi-label du sujet juridique
- **Source** : textes législatifs **de l'Union Européenne** (règlements, directives)
- **Langues** : 23 langues officielles EU (bg, cs, da, de, el, en, es, et, **fr**, fi, ga, hr, hu, it, lt, lv, mt, nl, pl, pt, ro, sk, sl, sv)
- **Taille** : ~65 000 documents par configuration
- **Configurations** : `level_1`, `level_2`, `level_3` (hiérarchie de classification à 3 niveaux de granularité)
- **Type de français** : législation européenne **traduite en français** (pas du droit FR natif)

### 2. Swiss Judgment Prediction

- **Tâche** : prédiction du jugement (accepté / rejeté)
- **Source** : **arrêts du Tribunal fédéral suisse**
- **Langues** : allemand, **français**, italien (les 3 langues officielles du tribunal)
- **Taille** : ~85 000 arrêts au total (toutes langues confondues)
- **Type de français** : arrêts rédigés en FR **par le Tribunal fédéral suisse** (droit suisse)

### Ce qui n'est PAS dans LEXTREME
- ❌ Aucun dataset de droit français national (Cassation, Conseil d'État, Conseil constitutionnel)
- ❌ Aucun dataset Légifrance
- ❌ Aucun dataset Judilibre
- ❌ Aucun dataset de QA ou génération

> [!warning] Couverture FR limitée
> LEXTREME inclut le français mais **pas le droit français national**. Les datasets FR disponibles = législation EU traduite + droit suisse. Pour du benchmarking sur Cassation/Légifrance, LEXTREME ne suffira pas.

## Methodologie

- Tâches de classification juridique supervisées
- Évaluation multi-lingue avec modèles XLM-R, mBERT, MiniLM
- Scores agrégés pour gérer la variance entre langues

## Resultats cles

| Modèle | Score agrégé |
|---|---|
| XLM-R large | 61.3 |
| mBERT | < 61.3 |
| XLM-R base | < 61.3 |

> Aucun modèle ne sature le benchmark → marge de progression réelle.

## Points forts

- **Seul benchmark NLP juridique multilingue à large couverture**
- Inclut le français (contrairement à LegalBench et LexGLUE)
- Méthodologie d'agrégation équitable (par lang + par dataset)
- HuggingFace + Weights & Biases publics

## Limites

- **Tâches essentiellement de classification** — pas de QA, pas de RAG, pas de génération
- Couverture FR limitée aux données européennes multilingues
- Pas de dataset spécifiquement sur le droit FR national (Cassation, Légifrance, etc.)
- Pas de raisonnement multi-saut

## Pertinence pour mon projet

> [!important] Le benchmark de référence pour le volet multilingue/FR
> C'est le **seul benchmark existant** qui permet de comparer des modèles sur du juridique FR dans un cadre standardisé. À étudier en priorité pour identifier les datasets FR disponibles et les performances actuelles. **MAIS** : ses tâches restent de la classification — insuffisant pour notre besoin de QA/raisonnement.

### Ce qu'on peut reutiliser
- Les datasets FR identifiés (cf. section dédiée)
- La méthodologie d'agrégation de scores inter-langues
- Les performances de référence (XLM-R, mBERT…) comme baseline à battre

### Ce qui manque pour notre projet
- **Tâches de raisonnement génératif** (QA, multi-saut) → on devra les construire
- **Corpus national FR** (Cassation, Légifrance) → on doit le constituer
- **Évaluation de la traçabilité** → métrique à créer

## Connexions
- [[Chalkidis-2022-LexGLUE]] — ancêtre monolingue
- [[Guha-2023-LegalBench]] — complément focus raisonnement vs classification
- [[Harvard-LIL-2024-Open-French-Law-RAG]] — benchmark FR spécifique
- [[COLD French Law Dataset]]

## Questions ouvertes

- [ ] Quels sont précisément les datasets FR dans LEXTREME ? (cf. fetch en cours)
- [ ] Peut-on utiliser le *split FR* de LEXTREME comme baseline directe pour nos modèles ?
- [ ] Y a-t-il un dataset LEXTREME spécifique à la jurisprudence FR ?
