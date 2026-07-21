---
date: 2026-07-21
type: decision
status: approved
tags: [decision, benchmark, k-fold, g1, g6, g7, lightgcn, ppr]
---

# Comparaison propre G1, G6 et G7 — Design

## Décision

La prochaine comparaison des graphes suit l'option A : réparer le protocole de validation avant de compléter la matrice de résultats. Les scores exploratoires déjà calculés restent archivés, mais ne sont pas utilisés pour choisir les nouvelles configurations.

## Objectif

Comparer causalement la base citation propre `G1`, les graphes typés `G6` et les huit variantes pondérées `G7`, puis expliquer pourquoi la performance plafonne et quelles transformations ont le plus de potentiel.

## Données et séparation

- Train et tuning : `train_augmented_retrievable_strict`, 5 603 questions.
- Évaluation interne : `eval_rich_retrievable_strict`, 754 questions.
- K-fold : 5 folds partagés par tous les graphes et toutes les méthodes.
- Groupe indivisible : composante connexe formée par les questions partageant `(source, doc_id, section_id)` ou le même énoncé normalisé.
- Un groupe entier est affecté à un seul fold.
- La répartition cherche à équilibrer le nombre de questions, le nombre de réponses attendues Articles et JP, le type de question et la granularité.

Cette règle remplace les folds actuels, qui stratifieraient seulement selon le nombre d'Articles et de JP et laissent des familles de reformulations traverser plusieurs folds.

## Méthodes et matrice minimale

### Contrôles

- Cosine direct Articles et JP : invariant au graphe, affiché comme baseline commune.
- `G1` : contrôle citation propre.
- `G6-citation-AA-knn5` et `G6-citation-JJ-knn5` : contrôles typés pondérés par cosine.

### Variantes G7

- Blocs : Article--Article et JP--JP.
- Poids `(citation, sémantique)` : `(1, 0.25)`, `(1, 0.50)`, `(1, 1.00)`, `(0.25, 1.00)`.
- kNN fixé à 5 pour cette première ablation ; il ne varie pas en même temps que les poids.

### PPR

- `k_in` dans `{5, 10, 20, 50}`.
- graines dans `{art_only, jp_only, both}`.
- `alpha` dans `{0.50, 0.70, 0.85, 0.95}`.
- Sélection distincte pour la cible Articles et la cible JP sur la moyenne des 5 folds.

### LightGCN

- Propagation `K` dans `{1, 2, 3}`.
- seeds `{42, 43, 44}` pour la shortlist ; un seed unique est accepté uniquement au stade de criblage.
- `lr` dans `{0.0005, 0.001}`.
- ancrage BGE `lambda_anchor` dans `{0.5, 1.0}`.
- négatifs `random` pour la comparaison principale ; les variantes de negative mining sont une ablation séparée.
- Le meilleur epoch est choisi sur chaque fold de validation. Le nombre d'epochs du replay est figé par agrégation des folds avant l'évaluation interne.

## Sélection et publication

- Cible Articles : classement primaire par moyenne `Recall@10`, puis `NDCG@10`, puis `MRR@10`.
- Cible JP : classement primaire par moyenne `Hit@10`, puis `NDCG@10`, puis `MRR@10`.
- Toujours reporter moyenne, écart-type et intervalle de confiance par fold.
- Toujours reporter les deux modalités pour détecter un transfert de performance Articles vers JP ou inversement.
- Les résultats sur l'eval interne sont calculés une seule fois après verrouillage des configurations et epochs.
- Les affirmations finales de l'article nécessiteront ensuite une lockbox nouvellement tirée et jamais inspectée.

## Outil de comparaison

L'artefact final comporte quatre vues liées :

1. matrice complète graphe × méthode × hyperparamètres, avec statut `CV`, `eval interne`, `exploratoire` ou `non exécuté` ;
2. champions CV Articles et JP, avec moyenne, dispersion et rang ;
3. courbes d'influence des hyperparamètres et courbes loss/validation de LightGCN ;
4. diagnostic des erreurs et du plafond de performance.

Les tableaux de la présentation sont générés à partir de ces mêmes artefacts, sans chiffres recopiés manuellement.

## Diagnostic du plafond

Pour chaque graphe, méthode et modalité, l'analyse mesure :

- couverture : part des réponses attendues encore présentes dans le graphe et dans le pool d'embeddings ;
- rang : distribution du rang de la première réponse attendue et du pourcentage de réponses attendues retrouvées pour `k` de 5 à 1 000 ;
- topologie : distance entre graines et réponses attendues, degré, communauté et type de relation emprunté ;
- généralisation : écarts train/validation, courbes de loss, epoch optimal et variance entre folds ;
- complémentarité : questions gagnées ou perdues par G1, G6 et G7, et recouvrement de leurs erreurs ;
- difficulté : résultats par nombre de réponses attendues, type de question, granularité et fréquence des nœuds attendus dans le train.

## Hypothèses à départager

1. Le principal plafond vient du ranking : les réponses existent mais restent trop loin du top-10.
2. Les liens sémantiques ajoutent des quasi-voisins, mais leur masse dilue le signal citation si les types ne sont pas appris séparément.
3. Les annotations gold sont incomplètes : les hard negatives cosine contiennent des quasi-positifs et dégradent BPR.
4. La supervision JP est insuffisante ou déséquilibrée, ce qui limite les gains JP de LightGCN.
5. La propagation homogène de LightGCN est trop simple pour exploiter plusieurs types d'arêtes ; un modèle relationnel ou un MLP ne se justifiera qu'après confirmation de cette hypothèse.

## Critère de sortie

Le chantier est terminé quand les folds groupés sont audités sans fuite, que PPR et LightGCN ont des résultats CV comparables sur G1/G6/G7, que les champions sont rejoués à configuration figée sur l'eval interne, et que l'outil diagnostique permet d'attribuer le plafond à la couverture, au ranking, à la topologie, à la supervision ou aux annotations.
