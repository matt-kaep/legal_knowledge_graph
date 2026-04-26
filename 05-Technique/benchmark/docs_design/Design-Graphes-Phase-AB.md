---
tags: [design, graphe, judilibre, rapprochements, articles, networkx, phase-ab]
type: design-document
status: livré
created: 2026-04-21
modified: 2026-04-21
---

# Design des graphes Phase A + Phase B (Cour de cassation)

> Deux phases consécutives construites le 21 avril 2026 à partir de la base
> Judilibre enrichie ([[Format-Fondement-Juridique]]). Complète [[Design-Benchmark-Rapprochements]].

---

## Phase A — Graphe unimodal JP → JP

### Schéma

- **Nœud** : un arrêt de la Cour de cassation.
  Clé primaire : **`pourvoi_norm`** (numéro de pourvoi normalisé `XX-XXXXX`).
  Attributs : `id_judilibre`, `ecli`, `chamber`, `chamber_raw`, `date`, `solution`, `publication`, `is_benchmark_src`, `shadow`.

  Le nœud est **`shadow`** quand il est cible d'un rapprochement mais n'apparaît pas comme source dans le corpus scanné (ex. arrêts pré-1980 non indexés).

- **Arête** (dirigée) : `rapproche(X, Y)` = l'arrêt X cite Y dans son champ `rapprochements` Judilibre.
  Attribut : `rel="rapproche"`, `weight`=nombre d'occurrences (>1 rare).

### Périmètres produits

Tous écrits dans `data/graphs/` avec pickle NetworkX et GraphML.

| Périmètre | Nœuds | Arêtes | Composantes | Plus grosse CC |
|---|---:|---:|---:|---:|
| **resserre** (1 532 sources benchmark + cibles directes) | 5 618 | 5 436 | 884 | 120 |
| **large** (tout arrêt ayant ≥1 rapprochement, source ou cible) | 21 534 | 18 201 | 5 199 | 1 725 |
| **tout_cc** (553 k arrêts, même sans lien) | 534 600 | 18 201 | 518 265 | 1 725 |

### Observations structurelles

- **Degré entrant max = 13** : pas de super-hubs. Les arrêts de principe ne dépassent pas la dizaine de citations institutionnelles.
- **Graphe très sparse** : densité ~1e-4. Cohérent avec la parcimonie des rapprochements publiés.
- **Fragmentation extrême** : 5 199 composantes disjointes (Phase A large). La Cour tisse des **lignées locales thématiques**, pas un réseau transversal.

### Top 5 arrêts-pivots (PageRank non dirigé sur large)

| pourvoi | Chambre | Date | Degré |
|---|---|---|---:|
| 07-20965 | Civ. 3e | 2009-09-23 | 13 |
| 13-19582 | Civ. 3e | 2014-12-17 | 12 |
| 14-18821 | Soc. | 2015-11-25 | 13 |
| 09-71734 | Civ. 3e | 2011-01-26 | 12 |
| 15-19973 | Soc. | 2017-03-22 | 10 |

---

## Phase B — Graphe bipartite JP × Articles

### Schéma

- **Nœuds** de type `decision` : identiques à Phase A (clé `pourvoi_norm`).
- **Nœuds** de type `article` :
  - Clé primaire : **`pair_key`** au format **`code_slug:article_num`** (ex. `code_civil:1240`, `code_du_travail:L122-14-3`).
  - Attributs : `code_slug`, `article_num`, `n_citations_corpus`.
  - Format issu de `enrichissement_base_complete.ipynb` (normalisation appliquée par `normalize_code` + `normalize_article`).

- **Arêtes** (dirigées) :
  - `rel="rapproche"` (Decision → Decision) — héritée Phase A.
  - `rel="cite"` (Decision → Article) — issue du champ `code_article_pairs`.
  - Dédup intra-arrêt : un arrêt qui cite `code_civil:1240` trois fois dans son texte produit **une seule arête** (pas de poids gonflé).

### Périmètres produits

Tous écrits dans `data/graphs_bipartite/`.

| Périmètre | Nœuds | JP | Articles | Arêtes `cite` | Arêtes `rapproche` | Composantes | Plus grosse CC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **resserre** | 9 593 | 5 618 | 3 975 | 18 366 | 5 958 | 16 | 9 586 (99,9 %) |
| **large** | 31 474 | 21 534 | 9 940 | 73 168 | 18 201 | 31 | 31 390 (99,7 %) |
| **tout_cc** | 567 943 | 534 600 | 33 343 | 1 196 681 | 18 201 | quasi-connexe | 493 721 (92 %) |

### Observation cruciale

**Les articles sont les ponts qui connectent le tissu jurisprudentiel.**
Entre Phase A et Phase B sur le périmètre large :
- Composantes : **5 199 → 31** (fragmentation divisée par 170).
- Plus grosse CC : 8 % → 99,7 % des nœuds.

Deux arrêts qui ne se citent pas explicitement peuvent être reliés en 2 sauts par un article commun. **C'est structurellement ce qui rend le KG utile au-delà d'un RAG vectoriel.**

### Top articles hubs

| pair_key | Citations dans le corpus | Domaine |
|---|---:|---|
| `code_de_procedure_civile:700` | 12 268 | Frais irrépétibles (universel) |
| `code_de_procedure_civile:455` | 2 837 | Motivation des jugements |
| `code_de_l_organisation_judiciaire:R431-5` | 2 553 | Composition des formations |
| `code_civil:1134` | 1 992 | Force obligatoire des contrats (avant 2016) |
| `code_civil:1382` | 1 030 | Responsabilité délictuelle (avant 2016) |

### Distribution des articles par code (top 15 sur large)

| Code | Articles distincts |
|---|---:|
| `code_du_travail` | 2 182 |
| `code_civil` | 1 316 |
| `code_de_procedure_penale` | 1 009 |
| `code_de_procedure_civile` | 819 |
| `code_de_la_securite_sociale` | 740 |
| `code_de_commerce` | 688 |
| `code_penal` | 455 |
| `code_rural_et_de_la_peche_maritime` | 289 |
| `code_de_la_sante_publique` | 267 |
| `code_de_la_consommation` | 243 |
| `code_des_assurances` | 215 |
| `code_des_procedures_civiles_d_execution` | 157 |
| `code_de_la_propriete_intellectuelle` | 154 |
| `code_monetaire_et_financier` | 124 |
| `code_general_des_impots` | 118 |

---

## Question méthodologique ouverte : renumérotation d'articles

Le graphe expose un **problème d'identité temporelle** sur les articles du Code civil :

| Ancien (avant réforme 2016) | Nouveau (depuis 2016) | Contenu |
|---|---|---|
| `code_civil:1382` | `code_civil:1240` | Responsabilité délictuelle |
| `code_civil:1383` | `code_civil:1241` | Faute ou négligence |
| `code_civil:1134` | `code_civil:1103`, `1104`, `1193` | Force obligatoire du contrat (éclaté) |
| `code_civil:1147` | `code_civil:1231-1` | Responsabilité contractuelle |

Dans le graphe actuel, ces paires sont des **nœuds distincts**. Conséquence :
- Un arrêt de 2010 citant `1382` et un arrêt de 2020 citant `1240` ne sont pas connectés via cet article, alors qu'ils portent sur la **même règle**.
- La centralité de `1382` est sous-estimée (car elle devrait cumuler avec `1240`).

**Options** (à décider pour le KG final) :
- **(a) Fusion canonique** : mapper `1382 → 1240` dans la normalisation. Plus cohérent juridiquement, mais perd la fidélité historique.
- **(b) Conservation + arête d'équivalence** : ajouter des arêtes `equivalent_to` entre anciens/nouveaux articles. Préserve la donnée brute et permet la traversée.
- **(c) Double indexation** : chaque citation est indexée à la fois par numéro historique et canonique (redondance). Plus coûteux mais parfaitement lisible.

**Recommandation initiale** : **(b)** pour le KG, car conserve la traçabilité et laisse au consommateur (RAG, GraphRAG) le choix de traverser ou non l'équivalence selon la date de l'arrêt.

→ à trancher collectivement, voir [[Format-Fondement-Juridique]].

---

## Pipeline et coût

### Temps d'exécution (M1)

| Étape | Temps |
|---|---|
| `build_rapprochement_graphs.py` (Phase A, 3 périmètres) | ~30 s |
| `build_bipartite_graphs.py` (Phase B, 3 périmètres) | ~45 s |
| `visualize_graphs.py` (6 PNG) | **~18 min** (spring_layout sur gros graphes) |

### Taille des artefacts

| Fichier | Taille |
|---|---:|
| `data/graphs/rapp-*.pkl` | 0,9-74 Mo |
| `data/graphs/rapp-*.graphml` | 2-8 Mo |
| `data/graphs_bipartite/bip-*.pkl` | 2,7-165 Mo |
| `data/graphs_bipartite/bip-*.graphml` | 6-23 Mo (pas de tout_cc, trop lourd) |
| `data/figures/*.png` | 0,7-6,8 Mo chacun |

Total disque : ~300 Mo.

---

## Prochaines étapes

1. **Ouvrir `bip-large.graphml` dans Gephi** — exploration manuelle + export visuel avancé.
2. **Détection Louvain** sur le bipartite — étiqueter chaque cluster par ses articles dominants → "thèmes juridiques latents".
3. **Projection unimodale article-article** via co-citation — article X et article Y co-cités dans N arrêts → lien induit de force N.
4. **Phase C — Benchmark enrichi** articulant rapprochements + articles (Q = arrêt, GT = {rapprochements officiels, articles cités} avec pondération par centralité).
5. **Résolution renumérotation** — trancher (a)/(b)/(c) et implémenter.

---

## Connexions

- [[2026-04-21]] — journal des livraisons
- [[Design-Benchmark-Rapprochements]] — benchmark utilisant les rapprochements
- [[Format-Fondement-Juridique]] — format canonique articles (question renumérotation)
- [[Format-Jurisprudence]] — format canonique JP
- [[Benchmark-KG-Juridique-FR-Design]] — vue d'ensemble
- Script Phase A : `build_rapprochement_graphs.py`
- Script Phase B : `build_bipartite_graphs.py`
- Script Viz : `visualize_graphs.py`
