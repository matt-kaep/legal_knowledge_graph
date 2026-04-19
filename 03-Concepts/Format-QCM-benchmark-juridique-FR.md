---
tags: [concept, benchmark, idee]
type: idee-format
status: "a-explorer"
created: 2026-04-16
modified: 2026-04-16
---

# Format QCM pour le benchmark juridique FR

> Inspiration : [[Zheng-2021-CaseHOLD]] (Stanford, 2021)

## Principe

Reprendre le **format QCM 1-correct / 4-distracteurs** de CaseHOLD, mais en le transposant au droit français et à la jurisprudence de la Cour de cassation.

## Pourquoi ce format

- **Évaluation automatique** simple (accuracy, F1)
- **Difficulté réglable** via le choix des distracteurs (plus ils sont plausibles, plus la tâche est difficile)
- **Tâche juridiquement réaliste** — c'est ce qu'un juriste fait : choisir le bon fondement parmi plusieurs possibles
- **Compatible avec tous les LLMs** (pas besoin d'évaluateur humain ni de LLM-as-judge)
- **Constructible automatiquement** à partir du KG (les distracteurs = voisins dans le graphe)

## Variantes de tâches envisageables

### V1 — Retrouver le fondement juridique

> Pour ce moyen, trouve l'article du code applicable parmi 5 candidats.

- Correct : l'article effectivement cité dans l'arrêt
- Distracteurs : 4 articles plausibles (même code, même chapitre, thèmes voisins)

### V2 — Retrouver la jurisprudence pertinente

> Pour cet argument/moyen, trouve l'arrêt de principe pertinent parmi 5 candidats.

- Correct : l'arrêt de référence cité par les juristes
- Distracteurs : 4 arrêts de la même juridiction, période proche, sujet connexe

### V3 — Détecter le revirement

> Cet arrêt casse/confirme-t-il un précédent ? Si oui, lequel parmi 5 ?

- Correct : l'arrêt effectivement cassé/confirmé
- Distracteurs : 4 arrêts antérieurs sur le même sujet

### V4 — Identifier le domaine juridique

> Cet arrêt relève de quelle branche du droit parmi 5 (pénal, civil, commercial, social, admin) ?

- Correct : la branche effective
- Distracteurs : les 4 autres branches principales

## Génération automatique des distracteurs

Le KG peut servir à **générer automatiquement** des distracteurs plausibles :

- Pour V1 (article) : voisins dans le graphe de citations (articles souvent cités ensemble)
- Pour V2 (arrêt) : arrêts de la même chambre, période proche, article pivot commun
- Pour V3 (revirement) : arrêts du même domaine dans une fenêtre temporelle

> Avantage : la difficulté des distracteurs mesure en elle-même la richesse du graphe.

## Difficulté graduée

| Niveau | Construction des distracteurs |
|---|---|
| Facile | Arrêts/articles aléatoires |
| Moyen | Arrêts/articles du même domaine général |
| Difficile | Voisins directs dans le graphe de citations |
| Très difficile | Arrêts sur le même point de droit mais avec solution différente |

## Limites à anticiper

- **Plus d'un bon candidat** : plusieurs articles peuvent fonder un même moyen → prévoir du **multi-label** ou "top-k correct"
- **Formulations alternatives** : une règle peut être citée via son n°, son nom, son texte → normaliser en amont
- **Biais de génération** : si les distracteurs sont générés par LLM, ils peuvent être trop faciles ou trop bizarres — validation humaine sur un échantillon

## Prochaines étapes

- [ ] Construire un MVP V1 (retrouver l'article) sur 100 arrêts Judilibre
- [ ] Définir la taxonomie des distracteurs
- [ ] Valider la calibration de difficulté sur 3 niveaux
- [ ] Comparer les performances des 4 baselines sur ce QCM

## Connexions

- [[Zheng-2021-CaseHOLD]] — source d'inspiration
- [[Guha-2023-LegalBench]] — cadre IRAC
- [[Harvard-LIL-2024-Open-French-Law-RAG]] — benchmark FR existant, complémentaire
- [[2026-04-14]] — réu superviseur : benchmark prioritaire
