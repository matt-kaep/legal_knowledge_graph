---
tags: [article, recommandation, gnn, reference]
categorie: "Recommandation"
titre_complet: "Solution_lightGCN.ipynb — exercice de référence LightGCN (Gowalla)"
auteurs: "Fourni par Johnny (Mattermost, #32)"
annee: 2026
type: "Notebook / exercice pédagogique"
venue: ""
url: "https://github.com/huangtinglin/NGCF-PyTorch (données Gowalla)"
status: "lu"
pertinence: "haute"
created: 2026-06-08
modified: 2026-06-08
---

# Solution_lightGCN.ipynb — implémentation de référence (Johnny, #32)

> [!info] Metadonnees
> **Source** : notebook envoyé par Johnny sur Mattermost (dépendance #32 du [[handoff-LightGCN]], désormais levée).
> **Contenu** : implémentation **minimale et propre** de LightGCN sur le dataset **Gowalla**, en PyTorch pur (pas de PyTorch Geometric).
> **Résultat** : **Recall@20 = 0,1843** (50 epochs) — cohérent avec le papier He et al. (~0,18 sur Gowalla).

## Résumé

C'est l'implémentation **canonique de référence** que Johnny considère comme le point de départ. Elle vaut surtout comme **squelette directement réutilisable** pour notre `31_lightgcn_train.py` : modèle, normalisation, BPR, boucle d'entraînement, éval. Données : 29 858 users × 40 981 items, 46 178 interactions (train.txt/test.txt de NGCF-PyTorch).

## Structure de l'implémentation (le squelette à reprendre)

### Modèle (`LightGCN(nn.Module)`)
- `user_embedding`, `item_embedding` = `nn.Embedding`, **init Xavier uniform**.
- `forward(graph)` :
  1. `all_emb = [cat(user_emb, item_emb)]` (empile users 0..U-1, items U..U+I-1) ;
  2. `num_layers` fois : `new = torch.spmm(graph, all_emb[-1])` (propagation creuse) ;
  3. `final = sum(all_emb) / (num_layers + 1)` (**layer combination = moyenne, couche 0 incluse**) ;
  4. re-split en `(user_emb, item_emb)`.
- `bpr_loss(users, pos, neg, graph)` : `pos = (u*p).sum(1)`, `neg = (u*n).sum(1)`, `loss = -log(sigmoid(pos - neg)).mean()`.
  → ⚠️ **pas de terme L2 explicite** : la régul passe par `AdamW(weight_decay=1e-4)`.

### Graphe (`create_adj_matrix`)
- Matrice **(U+I)×(U+I)** symétrique : pour chaque interaction `(u,i)`, on ajoute `u→(i+U)` **et** `(i+U)→u`.
- **Normalisation symétrique** `D^{-1/2} A D^{-1/2}` (gestion divide-by-zero), puis conversion en `torch.sparse`.

### Données BPR (`BPRDataset`)
- 1 négatif/positif, **rejection sampling** (re-tire tant que le négatif ∈ positifs du user). `batch_size = 2048`.

### Éval (`evaluate_recall`)
- `user_emb, item_emb = forward(graph)` ; scores `u @ item_emb.T` ; `argsort` top-K ; **Recall@20** moyenné.

### Hyperparamètres effectifs
| | Notebook | Papier | Notre fiche [[LightGCN-2020]] |
|---|---|---|---|
| dim | 64 | 64 | 1024 (init BGE-M3) |
| couches | 3 | 3 | 3 |
| optim | **AdamW** | Adam | — |
| lr | **0,01** | 1e-3 | à valider |
| weight_decay | 1e-4 | λ=1e-4 (L2 dans loss) | — |
| batch | 2048 | 2048 | 2048 |
| epochs | 50 | ~1000 (early stop) | 100 + early stop |

## ⚠️ LE choix de design que ce notebook met en lumière

Dans le notebook, **le graphe sur lequel on propage EST le graphe d'interactions user–item** (`train.txt`). Les users **ont des arêtes** et **propagent**. C'est le LightGCN canonique.

Or notre [[handoff-LightGCN]] prévoit l'inverse : questions **sans arête** à l'entraînement, propagation sur le graphe de **citations Art↔JP**. → trois designs possibles pour le script 31, à trancher (avec Johnny) :

| Design | Graphe de propagation | Avantage | Inconvénient |
|---|---|---|---|
| **A — Handoff** | Art↔JP (citations) ; questions = users **sans arête** (init BGE-M3 figée) | exploite nos 642k citations ; questions inductives | non-canonique ; questions ne propagent pas ; supervision = GT question→art |
| **B — Notebook-fidèle** | Question↔Article (GT train, comme `train.txt`) | **exactement** le notebook ; canonique | **n'utilise PAS le graphe de citations** ; GT épars + held-out |
| **C — Tripartite** | Question↔Article **∪** Article↔JP | utilise **tout** le signal ; le plus puissant | plus du LightGCN vanilla ; demande un graphe hétérogène soigné |

**Recommandation** : démarrer par **A** (fidèle au handoff, exploite notre actif unique = le graphe de citations) avec le squelette du notebook adapté, et garder **C** comme variante Session 4. **B** seule jette notre meilleur atout.

## Ce qu'on réutilise vs ce qu'on adapte (script 31)

**Réutilisé quasi tel quel** : classe `LightGCN`, `create_adj_matrix` (norm symétrique + `spmm`), `BPRDataset`, boucle BPR/AdamW.

**À adapter** :
- **Données** : nos `graph_penal.npz` (CSR JP×art) au lieu de `train.txt` ; restreindre au sous-graphe **vivant** (~46k articles, cf. [[ADR-001-Versionnage-Graphe-G0-Vn]] V1).
- **Init** : `nn.init.xavier_uniform_` → **charger les embeddings BGE-M3** (`emb_articles_all.npy`, `emb_jp_synthese.npy`, `questions_977_emb.npy`) ; gérer les nœuds non-embeddés (init aléatoire).
- **Éval** : remplacer le `Recall@20` maison par **notre `metrics.py`** (M1/M2/Hit/MRR/NDCG, `panel_strict_ext`) — éval strict + étendu, côté articles ET JP.
- **Design** : implémenter le mapping retenu (A/B/C ci-dessus).

## Connexions
- [[LightGCN-2020]] — le papier
- [[sota-gnn-reco-2026]] — mémo SOTA
- [[ADR-001-Versionnage-Graphe-G0-Vn]] — versionnage graphe + le design A/B/C à acter
- [[handoff-LightGCN]] — plan de sessions
