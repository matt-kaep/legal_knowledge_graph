---
tags: [article, benchmark, nlp-juridique, qcm]
categorie: "Benchmarks"
titre_complet: "When Does Pretraining Help? Assessing Self-Supervised Learning for Law and the CaseHOLD Dataset"
auteurs: "Zheng, Guha, Anderson, Henderson, Ho"
annee: 2021
type: "Article de conférence"
venue: "ICAIL 2021"
url: "https://arxiv.org/abs/2104.08671"
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# CaseHOLD — Zheng et al. 2021

> [!info] Metadonnees
> **Auteurs** : Zheng, Guha, Anderson, Henderson, Ho (Stanford)
> **Annee** : 2021 | **Venue** : ICAIL 2021
> **URL** : [Lien](https://arxiv.org/abs/2104.08671)

## Resume

Dataset de **53 000+ QCM** sur la prédiction de *holding* (doctrine/motif décisif) d'un arrêt cité. Source : 3.5M décisions de justice US. Tâche conçue comme **suffisamment difficile** pour bénéficier du pré-entraînement juridique spécialisé (+7.2% F1 vs BERT vanilla).

## Format de la tache — le point clé

Pour chaque question :
- **Contexte** : extrait d'un arrêt qui cite un autre arrêt
- **Question** : quelle est la *holding* (doctrine) de l'arrêt cité ?
- **5 candidats** : 1 correct + 4 distracteurs plausibles
- **Attendu** : sélectionner la bonne réponse

> C'est une tâche de **retrieval + compréhension** déguisée en QCM.

## Resultats cles

| Modèle | F1 |
|---|---|
| BiLSTM baseline | 0.40 |
| BERT | — |
| Legal-BERT (custom vocab) | +7.2% vs BERT (+12% relatif) |

## Points forts

- Tâche **juridiquement significative** (identifier la doctrine = compétence cœur du juriste)
- Échelle suffisante (53k) pour entraînement
- Prouve la valeur du pré-entraînement spécialisé
- **Format QCM** = métrique simple, évaluation automatique

## Limites

- 100% droit américain (case law)
- Anglais uniquement
- Notion de "holding" spécifique au common law (pas d'équivalent direct en civil law)

## Pertinence pour mon projet

> [!important] Format hautement transposable
> Le format **QCM "trouve la bonne parmi 5"** est le plus inspirant des benchmarks étudiés pour notre cas. Il offre évaluation automatique + difficulté réglable (via les distracteurs) + tâche juridiquement réaliste.

### Transposition en droit FR
Voir [[Format-QCM-benchmark-juridique-FR]] — note dédiée.

Exemples d'adaptations possibles :
- *"Pour ce moyen, trouve le fondement juridique (article) parmi 5 candidats"*
- *"Pour cet arrêt, trouve l'arrêt de revirement pertinent parmi 5 candidats"*
- *"Pour cette question de droit, trouve l'article du code pertinent parmi 5"*

### Ce qu'on peut reutiliser
- Le format QCM 1-correct-4-distracteurs
- La méthodologie de génération automatique des distracteurs (arrêts citant les mêmes articles, mais pas le bon holding)
- L'approche d'évaluation par F1 / accuracy simple

### Ce qu'il faut adapter
- Source : Judilibre + Légifrance au lieu de case law US
- Notion de "holding" → **fondement juridique** ou **principe applicable**
- Gérer la plurilingue (tout en FR)

## Connexions
- [[Guha-2023-LegalBench]]
- [[Chalkidis-2022-LexGLUE]] — CaseHOLD inclus comme tâche
- [[Format-QCM-benchmark-juridique-FR]]
