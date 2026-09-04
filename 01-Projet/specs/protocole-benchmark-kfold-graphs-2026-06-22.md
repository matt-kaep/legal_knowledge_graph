---
title: "Protocole benchmark K-fold sur graphes versionnés"
date: 2026-06-22
type: spec
tags: [spec, benchmark, graphs, g0, g1, g2, g3, kfold, metrics, lightgcn, ppr]
statut: "proposee"
---

# Protocole benchmark K-fold sur graphes versionnés

## 1. Objectif

Définir un protocole expérimental **strict, comparable et réutilisable** pour évaluer les méthodes de retrieval juridique sur les graphes `G0`, `G1`, `G2`, `G3`, avec :

- un **tuning équitable** des hyperparamètres ;
- une **comparaison graphe contre graphe à armes égales** ;
- un **eval final gelé** ;
- des **métriques homogènes** ;
- des **artefacts standards** pour les tableaux, slides et l'article.

Le protocole doit permettre de répondre honnêtement à deux questions distinctes :

1. **Quelle méthode est la meilleure sur un graphe donné ?**
2. **Quel graphe améliore réellement les performances, toutes choses égales par ailleurs ?**

---

## 2. Principe général

### 2.1 Jeux de données officiels

Le protocole officiel repose sur deux datasets uniquement :

- **train/tuning** : `train_augmented_retrievable_strict` — 5603 questions ;
- **eval finale** : `eval_rich_retrievable_strict` — 754 questions.

Ces deux jeux sont désormais les **seules références officielles** pour l'étape benchmark.

### 2.2 Découpage expérimental

Le rôle des datasets est figé :

- le **train strict** sert au **tuning** et à l'entraînement ;
- l'**eval strict** sert **uniquement** au score final publié.

### 2.3 Choix méthodologique principal

Le tuning ne se fait pas sur un simple split train/val fixe, mais via un **K-fold cross-validation** sur le `train_augmented_retrievable_strict`.

Choix retenu :

- **K = 5 folds**
- folds **déterministes**
- mêmes folds pour toutes les méthodes et tous les graphes

L'eval finale ne doit jamais être utilisée pour choisir un hyperparamètre.

---

## 3. Objets expérimentaux

Le protocole est le produit cartésien suivant :

`(version de graphe) × (famille de méthode) × (configuration d'hyperparamètres)`

### 3.1 Versions de graphe

- `G0` : graphe benchmark historique brut
- `G1` : `G0` sans articles isolés, sans articles abrogés / non `VIGUEUR`, puis sans JP devenues isolées
- `G2` : `G1` sans les 14 plus gros hubs articles
- `G3` : `G2` sans 15 articles procéduraux résiduels très généraux

### 3.2 Familles de méthodes

#### A. Baselines embeddings / cross-modal

- `B2-a` — cosine articles
- `B3-a` — cosine JP
- `B3-e` — articles via JP
- `B4-a` — cross-modal articles union
- `B4-c` — cross-modal JP union
- `B4-d` — cross-modal JP intersection
- `B4-e` — cross-modal JP RRF
- `B4-f` — cross-modal JP citation-weighted

#### B. Méthodes graphe non apprises

- `PPR` et ses variantes

#### C. Méthodes graphe apprises

- `LightGCN` propagé
- `LightGCN` entraîné

#### D. Méthodes LLM

- `LLM seul`
- `LLM + RAG`
- `LLM seul JP`
- `LLM + RAG JP`

### 3.3 Cibles évaluées

Le protocole principal publie deux cibles :

- **Articles stricts**
- **JP**

Le **dataset article étendu** n'est plus une cible principale. Il peut rester comme diagnostic secondaire, mais ne doit plus piloter le choix des champions ni l'histoire principale des résultats.

---

## 4. Règle d'équité entre graphes

### 4.1 Même benchmark de référence

La comparaison entre `G0`, `G1`, `G2`, `G3` doit rester **à question constante** :

- mêmes 5603 questions côté train ;
- mêmes 754 questions côté eval ;
- mêmes GT source de vérité.

On **ne redéfinit pas** le benchmark à chaque graphe.

### 4.2 Dénominateur GT figé

Pour comparer les graphes honnêtement, les métriques doivent être calculées avec le **même dénominateur GT de référence**.

Conséquence :

- si un graphe supprime un article ou une JP présents dans la GT, cette disparition doit se refléter dans la métrique ;
- on ne doit pas “améliorer” un graphe simplement parce qu'on a supprimé des cibles difficiles.

### 4.3 Couverture obligatoire à reporter

Comme certains graphes retirent des nœuds, chaque run doit reporter explicitement la **couverture benchmark** :

- `% d'occurrences article strict encore présentes`
- `% d'articles stricts uniques encore présents`
- `% de JP GT encore présentes`
- `% de questions dont au moins une GT article est encore atteignable`
- `% de questions dont au moins une GT JP est encore atteignable`

La couverture doit apparaître :

- dans les artefacts par graphe ;
- dans les tableaux de comparaison inter-graphes ;
- dans la présentation / l'article dès qu'un résultat est comparé entre graphes.

### 4.4 Interprétation scientifique

Une version de graphe n'est considérée meilleure que si :

1. elle améliore les métriques principales **ou**
2. elle garde des métriques comparables avec une structure bien plus propre

**tout en reportant son coût de couverture**.

---

## 5. Folds K-fold

## 5.1 Nombre de folds

- `n_folds = 5`

### 5.2 Déterminisme

Les folds doivent être figés une fois pour toutes via :

- un `random_state` fixe ;
- un fichier `fold_assignments.csv` versionné dans les artefacts benchmark.

### 5.3 Contraintes de construction des folds

Les folds doivent préserver autant que possible la structure du train :

- présence / absence de GT article ;
- présence / absence de GT JP ;
- niveau de richesse GT (1, 2, 3+ références) ;
- si possible, équilibre global des documents source.

### 5.4 Réutilisation

Les mêmes folds sont réutilisés pour :

- `G0`
- `G1`
- `G2`
- `G3`

Le graphe change ; **les folds ne changent pas**.

---

## 6. Panel de métriques officiel

## 6.1 Métriques publiées

Le panel officiel est :

- `Recall@10`
- `Hit@10`
- `MRR@10`
- `NDCG@10`
- `Normalized Rank`
- `LLM Judge`

### 6.2 Rôle des métriques

#### A. Métriques quantitatives principales

- `Hit@10`
- `MRR@10`
- `NDCG@10`
- `Recall@10`
- `Normalized Rank`

#### B. Métrique sémantique complémentaire

- `LLM Judge`

### 6.3 Usage méthodologique

#### Pour le tuning K-fold

Le tuning se fait **sans** `LLM Judge` comme objectif principal.

`LLM Judge` :

- coûte trop cher pour servir de métrique primaire de sweep ;
- reste une métrique d'analyse finale ;
- peut être calculé sur les champions retenus.

#### Pour l'eval finale

Le score final publié doit inclure :

- les 5 métriques quantitatives ;
- `LLM Judge` si disponible.

---

## 7. Règle de sélection des hyperparamètres

## 7.1 Principe

Chaque famille de méthode doit choisir ses hyperparamètres sur la **même logique** :

1. calculer les scores fold par fold sur le train strict ;
2. agréger les scores CV ;
3. choisir un champion via une règle fixe ;
4. rejouer ce champion sur l'eval stricte.

### 7.2 Règle de ranking des configurations

Pour chaque cible (`articles strict`, `JP`), les configurations sont classées par :

1. **métrique primaire : `Hit@10`**
2. **tie-break 1 : `NDCG@10`**
3. **tie-break 2 : `MRR@10`**
4. **tie-break 3 : `Recall@10`**
5. **tie-break 4 : `Normalized Rank`**

Cette règle est volontairement simple et commune.

### 7.3 Pourquoi `Hit@10` en primaire

`Hit@10` est retenu comme métrique primaire car :

- il reflète la part des bonnes réponses effectivement placées dans le top-10 ;
- il est comparable entre questions multi-GT ;
- il correspond bien au comportement retrieval attendu ;
- il évite la faiblesse du “présence binaire d'au moins un bon item”, déjà trop proche de l'ancien `M1`.

### 7.4 Un champion par cible

Une méthode peut avoir :

- un champion `articles strict`
- un champion `JP`

On n'impose pas qu'un même hyperparamètre soit optimal pour les deux modalités.

---

## 8. Hyperparamètres à tuner par famille

## 8.1 Baselines embeddings / cross-modal

### `B2-a`

Pas de vrai hyperparamètre structurel dans la version actuelle.

Statut :

- **pas de tuning majeur**
- sert de baseline fixe

### `B3-e`

Hyperparamètres :

- `k_in ∈ {5, 10, 20, 50}`

### `B4-a`

Hyperparamètres :

- `k_in ∈ {5, 10, 20, 50}`

### `B4-c`

Hyperparamètres :

- `k_in ∈ {5, 10, 20, 50}`

### `B4-d`

Hyperparamètres :

- `k_in ∈ {5, 10, 20, 50}`

### `B4-e`

Hyperparamètres :

- `k_in ∈ {5, 10, 20, 50}`
- éventuellement paramètre RRF si un jour exposé explicitement

### `B4-f`

Hyperparamètres :

- `k_in ∈ {5, 10, 20, 50}`

## 8.2 PPR

Hyperparamètres à tuner :

- `k_in ∈ {5, 10, 20, 50}`
- `seed_variant ∈ {art_only, jp_only, both}`
- `alpha ∈ {0.50, 0.70, 0.85, 0.95}`

Champion choisi séparément pour :

- articles strict
- JP

## 8.3 LightGCN

Le tuning LightGCN doit être fait par étapes, pour éviter l'explosion combinatoire.

### Étape A — tuning structurel de base

- `K ∈ {0, 1, 2, 3}`
- `seed ∈ {1, 2, 42}`

### Étape B — tuning entraînement

Pour le meilleur `K` ou les 2 meilleurs `K` :

- `lr`
- `epochs`
- `weight_decay` si exposé
- paramètres d'ancrage / régularisation si exposés

### Étape C — tuning data mining

Une fois le protocole de base stabilisé :

- stratégie de negative mining
- random negatives vs hard negatives
- hard negatives par similarité embedding

### Règle pragmatique

Le negative mining plus dur est un **niveau 2 de tuning** :

- pas nécessaire pour figer le protocole ;
- devient nécessaire pour améliorer proprement LightGCN après stabilisation du benchmark inter-graphes.

## 8.4 LLM et LLM + RAG

Pour la première version du protocole :

- **pas de K-fold tuning complet**
- **pas de prompt sweep systématique**

Règle :

- exécuter sur l'eval finale uniquement ;
- les inclure comme lignes comparatives dans les tableaux finaux.

Si une phase de prompt tuning est introduite plus tard, elle devra devenir un protocole séparé explicitement documenté.

---

## 9. Pipeline expérimental complet

Pour chaque graphe `G` dans la famille testée :

### Étape 1 — construire le bench du graphe

Produire pour `train` et `eval` :

- `bench_global.json`
- `stats.json`
- `questions_ids.npy`
- `questions_emb.npy`
- `coverage_summary.json`

### Étape 2 — mesurer la couverture

Calculer explicitement :

- couverture article stricte ;
- couverture JP ;
- questions entièrement / partiellement affectées.

### Étape 3 — charger les folds officiels

Réutiliser les mêmes folds K-fold pour tous les graphes.

### Étape 4 — K-fold par famille de méthode

Pour chaque configuration candidate :

1. entraîner / scorer sur `train_fold`
2. évaluer sur `val_fold`
3. stocker les métriques fold
4. agréger les 5 folds

### Étape 5 — choisir le champion

Appliquer la règle :

`Hit@10` > `NDCG@10` > `MRR@10` > `Recall@10` > `Normalized Rank`

### Étape 6 — relancer sur eval finale

Pour chaque champion retenu :

- exécuter une fois sur `eval_rich_retrievable_strict`
- produire les rankings complets

### Étape 7 — lancer `LLM Judge`

Sur l'eval finale uniquement, pour :

- champions embeddings / graphe
- LLM seul / RAG

### Étape 8 — construire les tableaux finaux

Produire :

- tableau articles strict
- tableau JP
- tableau comparaison inter-graphes

---

## 10. Artefacts standards à produire

## 10.1 Par graphe

Arborescence cible conceptuelle :

`data/doctrine_v3plus_bench/<graph_version>/<split_or_eval>/...`

Exemples :

- `data/doctrine_v3plus_bench/G0/train_augmented_retrievable_strict/`
- `data/doctrine_v3plus_bench/G0/eval_rich_retrievable_strict/`
- `data/doctrine_v3plus_bench/G1/...`
- `data/doctrine_v3plus_bench/G2/...`

## 10.2 Artefacts K-fold

Pour chaque méthode / famille :

- `fold_assignments.csv`
- `cv_results_raw.csv`
- `cv_results_summary.csv`
- `champions.json`

## 10.3 Artefacts eval finale

- `eval_m1_m2.csv`
- `rankings.parquet`
- `eval_m3.csv`
- `eval_m3_summary.json`
- `global_table_articles.csv`
- `global_table_jp.csv`
- `global_table_graph_comparison.csv`

## 10.4 Artefacts de narration

- tableaux slides
- figures tuning par méthode
- figures comparaison `G0/G1/G2/G3`

## 10.5 Commandes de référence G0

Les scripts du harnais K-fold inter-graphes sont maintenant branchés.

### Construction du protocole partagé

```bash
python 05-Technique/benchmark/etape1_embedding_pur/scripts/41_make_kfold_assignments.py --graph-version G0
```

### CV par famille

```bash
python 05-Technique/benchmark/etape1_embedding_pur/scripts/42_run_cv_b3_b4.py --graph-version G0
python 05-Technique/benchmark/etape1_embedding_pur/scripts/43_run_cv_ppr.py --graph-version G0
python 05-Technique/benchmark/etape1_embedding_pur/scripts/44_run_cv_lightgcn.py --graph-version G0
```

### Rejeu final des champions et tableaux

```bash
python 05-Technique/benchmark/etape1_embedding_pur/scripts/45_run_final_champions.py --graph-version G0
python 05-Technique/benchmark/etape1_embedding_pur/scripts/46_build_protocol_figures.py --graph-version G0
```

### Sorties attendues

- protocole partagé :
  - `data/doctrine_v3plus_bench/_protocol/train_augmented_retrievable_strict/fold_assignments.csv`
- sorties CV par famille :
  - `cv_results_raw.csv`
  - `cv_results_summary.csv`
  - `champions.json`
- sortie finale par graphe :
  - `final_champions_summary.csv`
  - `global_table_articles.csv`
  - `global_table_jp.csv`
  - `global_table_graph_comparison.csv`

### État actuel d'implémentation

- scripts `41` à `46` : implémentés ;
- validations locales :
  - tests unitaires/fonctionnels ciblés sur folds, CV wrappers, LightGCN history, replay final ;
  - test synthétique de bout en bout sur `45_run_final_champions.py --skip-replay` ;
- validation encore à exécuter :
  - rerun réel `G0` avec de vrais artefacts `champions.json` / `cv_results_summary.csv` produits par les runners.

---

## 11. Tableaux attendus dans la présentation / l'article

## 11.1 Tableau par méthode

Pour chaque méthode, prévoir une slide ou annexe avec :

- toutes les variantes testées ;
- tous les hyperparamètres testés ;
- toutes les métriques CV moyennes ;
- la ligne championne clairement identifiée ;
- un ranking interne des variantes basé sur `Hit@10`.

Ce tableau doit être séparé :

- côté articles strict
- côté JP

## 11.2 Tableau final par graphe

Pour chaque graphe :

- tableau final articles strict
- tableau final JP

Colonnes :

- méthode
- `Recall@10`
- `Hit@10`
- `MRR@10`
- `NDCG@10`
- `Normalized Rank`
- `LLM Judge`

## 11.3 Tableau de comparaison inter-graphes

Un tableau final doit comparer, pour chaque méthode championne :

- `G0`
- `G1`
- `G2`
- `G3`

avec :

- métriques finales ;
- couverture benchmark associée.

---

## 11.4 Figures attendues pour la présentation et l'article

Les tableaux ne suffisent pas. Chaque famille de méthode qui possède un tuning réel doit aussi produire des **figures de lecture**.

### A. Courbes d'influence des hyperparamètres

Pour chaque méthode avec sweep, produire au minimum une ou plusieurs courbes montrant l'effet des hyperparamètres principaux sur les métriques.

#### Baselines embeddings / cross-modal

Pour `B3-e`, `B4-a`, `B4-c`, `B4-d`, `B4-e`, `B4-f` :

- courbe ou barplot de `k_in -> Hit@10`
- idéalement aussi `k_in -> NDCG@10`

#### PPR

Pour `PPR`, produire :

- courbe `alpha -> Hit@10`
- courbe `k_in -> Hit@10`
- une figure séparée ou facettée par `seed_variant`

Si la place est limitée :

- `Hit@10` reste la courbe principale à montrer ;
- `NDCG@10` peut servir de courbe secondaire ou d'annexe.

#### LightGCN

Pour `LightGCN`, produire :

- courbe `K -> Hit@10`
- courbe `K -> NDCG@10`
- si plusieurs seeds sont testées, afficher la moyenne et la dispersion

### B. Courbes d'entraînement

Pour les méthodes apprises, produire systématiquement les courbes d'entraînement.

#### LightGCN

À minima :

- `epoch -> training loss`

Si possible également :

- `epoch -> validation Hit@10`
- `epoch -> validation NDCG@10`

Cela permet de voir :

- si la convergence est propre ;
- si le modèle continue à apprendre ;
- si l'on observe un sur-apprentissage ;
- si le meilleur nombre d'epochs est cohérent avec le choix final.

### C. Niveau de lecture attendu

Les figures doivent permettre de répondre visuellement à des questions simples :

- la méthode est-elle stable ou très sensible à son hyperparamètre ?
- le champion est-il un vrai optimum ou un point isolé fragile ?
- le graphe nettoyé modifie-t-il la forme de la courbe ?
- l'entraînement LightGCN converge-t-il mieux sur `G1/G2` que sur `G0` ?

### D. Comparaison inter-graphes

Quand le protocole sera réappliqué à `G0/G1/G2/G3`, les mêmes figures doivent pouvoir être comparées entre graphes.

En pratique :

- même axe x ;
- même métrique affichée ;
- même convention de couleurs ;
- même règle d'agrégation K-fold.

Cela permettra de montrer non seulement le meilleur score final, mais aussi si un graphe :

- rend la méthode plus performante ;
- rend le tuning plus stable ;
- réduit la variance ;
- améliore la dynamique d'entraînement.

### E. Priorité de production

Les figures à produire en priorité sont :

1. `PPR : alpha -> Hit@10`
2. `PPR : k_in -> Hit@10`
3. `B3-e/B4-* : k_in -> Hit@10`
4. `LightGCN : K -> Hit@10`
5. `LightGCN : epoch -> loss`
6. `LightGCN : epoch -> validation Hit@10`

Ces figures sont prioritaires pour la présentation, puis réutilisables dans l'article.

---

## 12. Règle d'application dans le temps

### 12.1 Première application

Le protocole sera d'abord appliqué au **premier graphe de référence**, c'est-à-dire :

- `G0` si l'objectif immédiat est de valider la mécanique K-fold ;
- ou `G1` si l'objectif immédiat est de repartir directement de la base propre minimale.

### 12.2 Réapplication aux autres graphes

Une fois la mécanique validée sur le premier graphe :

1. même code
2. mêmes folds
3. même panel
4. mêmes règles de sélection
5. seule la version de graphe change

Le protocole n'est considéré réussi que s'il est **ré-exécutable sans ambiguïté** sur `G0`, `G1`, `G2`, `G3`.

---

## 13. Garde-fous

### 13.1 Ce qu'il ne faut pas faire

- choisir un champion sur `eval_rich_retrievable_strict`
- retuner `G2` avec des règles différentes de `G1`
- changer les folds en fonction du graphe
- masquer la perte de couverture quand un graphe retire des GT
- comparer des champions tunés sur `G0` avec des versions non retunées sur `G1/G2/G3`

### 13.2 Ce qu'il faut toujours reporter

- version du graphe
- dataset
- folds utilisés
- grille d'hyperparamètres
- métrique primaire de sélection
- couverture benchmark

---

## 14. Décisions opérationnelles retenues

À ce stade, le protocole officiel retient :

- `5-fold CV` sur `train_augmented_retrievable_strict`
- `eval_rich_retrievable_strict` gelé pour les scores finaux
- `Hit@10` comme métrique primaire de sélection
- `NDCG@10`, `MRR@10`, `Recall@10`, `Normalized Rank` comme tie-breakers
- `LLM Judge` sur les champions finaux
- comparaison inter-graphes à dénominateur GT figé
- couverture benchmark reportée explicitement
- cible principale : `articles strict` + `JP`

---

## 15. Conséquence pratique immédiate

La prochaine étape de développement n'est pas de “lancer quelques scripts”, mais de brancher le repo sur ce protocole :

1. fichier de folds officiel ;
2. runners K-fold réutilisables ;
3. sélection standardisée des champions ;
4. sorties inter-graphes homogènes ;
5. tableaux finaux générés automatiquement.

Une fois cette couche en place, on pourra :

- l'appliquer d'abord sur le premier graphe choisi ;
- puis la réexécuter proprement sur `G1`, `G2`, `G3`.
