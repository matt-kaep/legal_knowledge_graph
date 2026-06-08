---
tags: [fiche, gnn, recommandation, synthese]
type: synthese
created: 2026-06-08
modified: 2026-06-08
pertinence: haute
sujet: Revue SOTA GNN pour reco/link-prediction — préparation prototype LightGCN
---

# SOTA GNN pour recommandation & prédiction de liens (2026)

> [!abstract] Objectif
> Mémo de revue (Session 1 du handoff `handoff-LightGCN.md`). But : avoir une vue **précise** des hyperparamètres typiques et des conventions d'évaluation avant d'implémenter LightGCN, et situer les variantes (R-GCN, HGT) pour la suite. Cible : insérer une ligne GNN dans le **grand tableau global** du Chantier 2, et **battre PPR sur ≥ 1 régime**.

## 1. Tableau comparatif

| Modèle | Année / venue | Tâche cible | Hétéro ? | Inductif ? | Hyperparams clés | Métriques rapportées | Force / Faiblesse |
|---|---|---|---|---|---|---|---|
| **LightGCN** | 2020 SIGIR | Reco (CF) biparti user–item | Non | Non | `K=3`, dim 64, `lr=1e-3`, `λ=1e-4`, BPR, batch 2048, layer-comb moyenne | Recall@20, NDCG@20 | **+** simple, rapide, baseline reco incontournable ; **−** pas de features, transductif |
| **NGCF** | 2019 SIGIR | Reco (CF) biparti | Non | Non | dim 64, `K=3`, `lr≈1e-3`, message+node dropout, `W` par couche + σ | Recall@20, NDCG@20 | **+** historique fondateur ; **−** sur-paramétré, battu par LightGCN (strawman) |
| **R-GCN** | 2018 ESWC | Link prediction & node classif sur KG **multi-relationnel** | **Oui** | Non | 2 couches, `W_r` par relation + **basis/block decomposition**, dropout, décodeur DistMult | **MRR, Hit@1/3/10** (link pred) ; accuracy (node) | **+** gère les types d'arêtes ; **−** explosion params si bcp de relations, ne scale pas seul |
| **HGT** | 2020 WWW | Graphes **hétérogènes** larges (OAG) | **Oui** | Non (transductif ; HGSampling = mini-batch/scalabilité, **pas** généralisation inductive) | dim 256–512, `K=2–4`, 8 têtes, attention type-spécifique (Q/K/V par type) | F1 / accuracy (node classif), MRR (link) | **+** forte capacité, attention type-aware ; **−** lourd, gourmand en données |
| **GraphSAGE** | 2017 NeurIPS | Node embedding **inductif** | Non (natif) | **Oui** | 2 couches, échantillons `S1=25, S2=10`, dim 128–256, agrégateur mean/pool/LSTM | F1 (node), AUC (link) | **+** inductif, scalable par sampling ; **−** homogène, l'agrégation peut perdre du signal |

### Conventions d'évaluation par famille
- **Reco / CF** (LightGCN, NGCF) : full-ranking sur tous les items non vus → **Recall@20 + NDCG@20** (parfois @10/@40). Un négatif par positif (BPR).
- **Link prediction sur KG** (R-GCN) : ranking par filtered-MRR + **Hit@K** (K=1,3,10). Négatifs par corruption (head/tail).
- **Node classification** (HGT, GraphSAGE) : **Micro/Macro-F1**.

→ Pour nous, le panel cible (M1, M2, Hit@K, MRR, NDCG) **recouvre les deux conventions reco et link-pred** : on est compatibles avec la littérature des deux familles.

## 2. Lecture stratégique pour le projet

### Ordre d'implémentation recommandé
1. **LightGCN homogène** (Art↔JP comme un seul type d'item) — fidèle au papier, mécanique = notre PPR mais apprise. **Baseline GNN n°1.**
2. **R-GCN** si LightGCN concluant — exploite les types d'arêtes (Art→JP « cité_par » vs Art↔Art) ; décodeur DistMult pour link-pred Art↔JP.
3. **HGT** seulement si capacité supplémentaire justifiée — attention hétérogène, coûteux, à réserver si R-GCN sature.
4. **GraphSAGE** : intérêt surtout pour l'**inductif** (nouvelles questions/articles non vus) — pertinent pour la généralisation, pas pour battre PPR au départ.

### Ablation décisive : LightGCN `K=0`
`K=0` (aucune propagation) = produit scalaire des embeddings d'ID = **notre baseline cosine BGE-M3**. Donc :
- `K=0` ≈ cosine pur ;
- `K=1,2,3` mesurent **exactement** ce que la propagation graphe ajoute au-dessus du sémantique.
C'est l'expérience qui isole « apport du graphe vs apport du texte » — argument central du mémoire. À instrumenter dès la première éval.

### Décision de split (rappel handoff, Option A retenue)
- **Train graph** = toutes les citations Art↔JP du graphe.
- **Train pairs** = questions doctrine_qgen hors cohorte.
- **Val / Test** = cohorte 971, jamais vue à l'entraînement (anti-leak).

## 3. Implications des données réelles (constats 2026-06-08)

> [!warning] Écarts mesurés vs hypothèses du handoff
> Vérifications faites sur `graph_penal.npz` + embeddings réels avant tout code.

| Élément | Hypothèse handoff | **Réalité mesurée** |
|---|---|---|
| Fichier graphe | `graph_v5.npz` dans `global_bench/` | **`baseline_b2/penal_bundle/graph_penal.npz`** (`config.GRAPH_NPZ`) |
| Dimensions | 118 112 art × 87 821 JP | **CSR [118 112 JP × 87 821 articles]**, 642 450 citations (axes inversés dans le handoff) |
| Emb articles | `articles_embeddings.npy` 87 821×1024 | **`emb_articles_all.npy` 31 357×1024** — seuls 36 % des articles du graphe embeddés |
| Emb JP | `jp_synthese_embeddings.npy` | **`emb_jp_synthese.npy` 116 755×1024** (99 % des JP) |
| Emb questions | `questions_977_emb.npy` 971×1024 | ✓ conforme |
| PyTorch Geometric | supposé dispo | **absent** du `.venv` (torch 2.12.0 + MPS OK) → install à valider |

**Couverture GT (pool 2674 questions, `bench_article*.json`)** :
- GT **strict** : 1312 articles uniques → 1174 dans le graphe, **1115 embeddés**, 59 dans-graphe-sans-emb, 138 hors-graphe.
- GT **étendu** : 2553 uniques → 2415 dans le graphe, **1966 embeddés**, 449 dans-graphe-sans-emb, 138 hors-graphe.
- Questions avec ≥1 GT non-embeddé : **101/2674 (strict)**, **385/2674 (étendu)**.

**Verdict** (corrigé) : la couverture partielle n'est **pas bloquante** pour les **GT** (seulement 9 articles GT morts, 7 questions strictes perdues sur la cohorte), mais le graphe G0 est **très bruité** — voir cartographie ci-dessous. Un nœud non-embeddé n'apprend par propagation **que s'il est cité** ; **41 824 articles (48 %) sont « doublement morts »** (ni texte ni citation) et restent inertes. Les 138 articles hors-graphe sont un **plafond partagé avec PPR**. → Traité comme **version G0** à nettoyer (cf. [[ADR-001-Versionnage-Graphe-G0-Vn]]).

#### Cartographie des 87 821 articles (axes texte × citation)
| | Embeddé | Non embeddé |
|---|---|---|
| **Cité** (degré ≥ 1) | 13 236 — signal max | 14 640 — apprend via propagation |
| **Jamais cité** (degré 0) | 18 121 — texte seul (= cosine) | **41 824 — doublement morts** |

**Autres constats qualité** : graphe **binaire** (`data` = 1, fréquence de citation perdue) ; degré article médiane 0, max 35 502 (très déséquilibré) ; sous-graphe vivant sain (JP→art moy 5,4). **Supervision JP** : ~7 questions seulement au train (971/978 questions à GT-JP sont dans la cohorte) → JP **non entraînable directement**, retrieval par transfert via les arêtes Art↔JP. Voir l'expérience d'augmentation (régime D) dans l'ADR.

## 4. Dépendances ouvertes (à lever avant Sessions 2–3)

- [ ] **#32 — Fichier Mattermost de Johnny** : non récupérable par l'agent → à fournir par l'utilisateur. Bloque **une fiche** (`02-Etat-de-l-art/recommandation/`), pas le train. Stub parking créé.
- [x] **Chantier 1 — panel métriques : FAIT** (Week-10, branche `etat-lieux-johnny-2026-05-28`). `scripts/metrics.py` expose M1 (Recall@K), M2 (rang custom), Hit@K, MRR@K (cappé à K), NDCG@K (rel binaire) + `all_metrics` / `panel_strict_ext`. Docstring : « Importé par 18 (B*), 20 (PPR), 23 (M3, futur), **31 (LightGCN, futur)** ». L'éval va jusqu'à `24_build_global_table.py`. → LightGCN doit **réutiliser ce `metrics.py` tel quel** (script 31), aucune métrique custom.
- [ ] **PyTorch Geometric** : `pip install torch_geometric` dans le `.venv` (pas l'env `--user` cluster, cf. mémoire `cluster-user-env-fragile.md`) → confirmation utilisateur requise (Session 2).
- [ ] **Décision init BGE-M3** : freeze N epochs puis fine-tune, vs fine-tune dès le départ avec `lr` réduit.

## 5. Références

- He et al. 2020, *LightGCN*, SIGIR — [[LightGCN-2020]]
- Wang et al. 2019, *NGCF*, SIGIR — arxiv 1905.08108
- Schlichtkrull et al. 2018, *R-GCN*, ESWC — arxiv 1703.06103
- Hu et al. 2020, *HGT*, WWW — arxiv 2003.01332
- Hamilton et al. 2017, *GraphSAGE*, NeurIPS — arxiv 1706.02216
- Connexes projet : [[Tang-2024-CaseLink-Inductive-Graph-Learning]], [[Wang-2022-Legal-Judgment-Heterogeneous-Graphs]], [[Wendlinger-2025-Joint-Legal-Citation-Prediction]]
