---
tags: [article, benchmark, nlp-juridique]
categorie: "Benchmarks"
titre_complet: "LexGLUE: A Benchmark Dataset for Legal Language Understanding in English"
auteurs: "Chalkidis, Jana, Hartung, Bommarito, Androutsopoulos, Katz, Aletras"
annee: 2022
type: "Article de conférence"
venue: "ACL 2022"
url: "https://arxiv.org/abs/2110.00976"
status: "lu"
pertinence: "moyenne"
created: 2026-04-16
modified: 2026-04-16
---

# LexGLUE — Chalkidis et al. 2022

> [!info] Metadonnees
> **Auteurs** : Chalkidis, Jana, Hartung, Bommarito, Androutsopoulos, Katz, Aletras
> **Annee** : 2022 | **Venue** : ACL 2022
> **URL** : [Lien](https://arxiv.org/abs/2110.00976)

## Resume

Premier benchmark NLP juridique unifié, inspiré de GLUE. Regroupe **7 tâches** de compréhension du droit en **anglais uniquement**. Vise à mesurer la généralisation des modèles NLU sur des tâches juridiques variées.

## Les 7 taches

| Tâche | Type | Source |
|---|---|---|
| ECtHR (Task A) | classification multi-label | Cour EDH |
| ECtHR (Task B) | classification multi-label | Cour EDH |
| SCOTUS | classification multi-class | Cour suprême US |
| EUR-LEX | classification multi-label | droit européen |
| LEDGAR | classification multi-class | clauses contractuelles |
| UNFAIR-ToS | classification multi-label | CGU |
| CaseHOLD | QCM | case law US |

## Langue

- **Anglais uniquement** → pas de FR

## Points clés

- Standardise l'évaluation NLP juridique (avant LegalBench)
- HuggingFace + GitHub publics
- **Saturé** aujourd'hui : les modèles dépassent les humains sur plusieurs tâches

## Pertinence pour mon projet

> [!note] Intérêt limité
> Pas de français, tâches essentiellement de classification — moins adapté à notre besoin QA/retrieval juridique. **Intérêt historique et méthodologique** : modèle de benchmark "à la GLUE" qu'on peut imiter structurellement.

### Ce qu'on peut reutiliser
- Le principe **benchmark unifié multi-tâches**
- La distinction claire entre classification, QCM, et génération

### Limites pour nous
- Anglais uniquement
- Pas de tâche RAG ni QA
- Focus classification — pas de raisonnement

## Connexions
- [[Guha-2023-LegalBench]] — successeur conceptuel
- [[CaseHOLD-Zheng-2021]] — inclus comme tâche
- [[LEXTREME-Niklaus-2023]] — extension multilingue de LexGLUE
