---
type: handoff
sujet: Prototype LightGCN sur le graphe biparti Art↔JP
sessions_estimees: 3-4 sessions (lecture + setup + train + eval) sur 2-3 semaines
date_creation: 2026-06-08
tasks_liees: [#31, #32, #33]
---

# Handoff — Prototyper LightGCN comme baseline GNN du tableau global

## Objectif global

Implémenter **LightGCN (He et al. 2020)** sur le graphe biparti Article↔JP du projet, l'entraîner, et l'évaluer sur le panel métriques complet (M1, M2, Hit, MRR, NDCG, M3) pour l'insérer dans le grand tableau global. **Critère de succès** : battre PPR sur au moins un régime (singleton ou multi-GT, côté articles ou JP).

LightGCN est la baseline GNN incontournable en reco system — Johnny l'a explicitement nommée Week-9. Conceptuellement : c'est une **PPR apprise** (propagation linéaire sans transformation non-linéaire), donc transition naturelle depuis notre baseline graphe non-apprise.

## Pré-requis bloquants (faire AVANT cette session)

1. **Récupérer le fichier Mattermost envoyé par Johnny** (#32). Lecture obligatoire, contient les références reco system qu'il considère canoniques. **Stocker dans** `02-Etat-de-l-art/recommandation/` avec fiche de lecture.
2. **Lecture du papier LightGCN** original (He et al. 2020, SIGIR) — fiche dans `02-Etat-de-l-art/gnn/`.
3. **Revue benchmarks SOTA** (#33) : NGCF, R-GCN, HGT, GraphSAGE — survol pour identifier les hyperparamètres canoniques et les métriques rapportées (typiquement Recall@20 + NDCG@20).
4. **Chantier 1 terminé** : panel métriques (Hit, MRR, NDCG) opérationnel via `metrics.py`. Sinon LightGCN ne pourra pas être évalué proprement.

## Contexte projet

- **Graphe biparti existant** : 118 112 articles × 87 821 JP, ~2M citations (642k actives après dédup). Stocké dans `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/` (cf. `config.GRAPH_NPZ`).
- **Embeddings BGE-M3 disponibles** : articles (1024-dim) + JP synthèses (1024-dim). Servent à **initialiser** les embeddings LightGCN au lieu d'aléatoire — accélère convergence et exploite le signal sémantique pré-existant.
- **Cohorte d'éval** : 971 questions doctrine_qgen + CRFPA, GT strict + GT étendu.
- **Baseline à battre** : PPR row-norm α=0,95 sur étendu (M1=0,441), B3-e sur strict (M1=0,711).

## Décisions architecturales déjà prises

- **Question = "utilisateur virtuel"** : on traite chaque question comme un nœud user dans un graphe tripartite (User=Question, Item=Article, Item=JP), initialisé par son embedding BGE-M3.
- **Pas de retrain à chaque question** : embeddings appris une fois, inference = lookup + propagation.
- **Évaluer sur le panel complet** côté articles ET côté JP.
- **Split temporel** : train/val/test sur les citations existantes — pas leak entre questions et GT.

## Session 1 — Lecture + revue SOTA (3-4h, sans code)

Pré-requis #32 et #33. Livrables :
- Fiche `02-Etat-de-l-art/gnn/LightGCN-2020.md`
- Fiche `02-Etat-de-l-art/recommandation/[fichier-mattermost].md`
- Court mémo `06-Analyses/syntheses/sota-gnn-reco-2026.md` : tableau Modèle × Métriques rapportées × Hyperparams clés × Forces/faiblesses (LightGCN, NGCF, R-GCN, HGT, GraphSAGE)

Objectif : avant la session 2, on doit avoir une vue **précise** des hyperparams typiques (nb couches, dim embeddings, λ régul, batch size, lr) et des conventions d'évaluation (split, négatifs, métriques).

## Session 2 — Setup data + scaffolding (4-5h)

### Étape 1 — Stack technique
- **PyTorch Geometric** (`torch_geometric`) : framework GNN standard. Vérifier compatibilité avec env actuel ; sinon créer venv dédié `legal-gnn-env`.
- **NE PAS** polluer l'env --user du cluster (cf. mémoire `cluster-user-env-fragile.md`).
- GPU recommandé pour le train ; CPU fallback possible mais ~10× plus lent.

### Étape 2 — Conversion graphe → PyG `HeteroData`
Script à créer : `05-Technique/benchmark/etape1_embedding_pur/scripts/30_build_pyg_graph.py`

Structure cible :
```
HeteroData(
  article = {x: [87821, 1024]},          # embeddings BGE-M3 articles
  jp      = {x: [118112, 1024]},         # embeddings BGE-M3 synthèses JP
  question= {x: [971, 1024]},            # embeddings BGE-M3 questions (cohorte)
  ('jp', 'cite', 'article')   = {edge_index: [2, ~2M]},
  ('article', 'cited_by', 'jp')= {edge_index: [2, ~2M]},  # reverse
  # PAS d'arêtes question -> article/jp à l'entrainement
  # (les GT sont positifs de test/val)
)
```

Décision à prendre : **graphe homogène bipartite** (article+JP comme un seul type d'item) vs **hétérogène** (types distincts). LightGCN canonique = homogène. Recommandation : commencer **homogène** sur Art↔JP pour rester fidèle au papier, garder hétérogène pour R-GCN/HGT.

### Étape 3 — Split train/val/test
Choix du split :
- **Option A** (recommandée) : split sur les **questions**, pas sur les citations. Train graph = toutes les citations Art↔JP. Train pairs = questions doctrine_qgen passées (hors cohorte 971). Val/Test = 971 questions cohorte split 50/50 ou 30/70.
- **Option B** : split temporel des citations (plus complexe, pas de gain à ce stade).

→ Aller Option A. Justifier dans le journal.

## Session 3 — Implémentation LightGCN + train (5-6h + compute)

### Étape 1 — Modèle LightGCN minimal
Architecture canonique :
```python
class LightGCN(nn.Module):
    def __init__(self, num_articles, num_jp, num_questions, emb_dim, n_layers):
        # embeddings = nn.Embedding (init by BGE-M3)
        # n_layers propagation linéaire (no MLP, no non-linearity)
        # final embedding = mean(emb_layer_0, ..., emb_layer_n)

    def propagate(self, edge_index):
        # symmetric normalization (DAD)
        # x_{l+1} = A_norm @ x_l

    def forward(self, q_idx, art_idx):
        # score = q_emb @ art_emb.T

    def bpr_loss(self, q, pos, neg):
        # -log(sigmoid(score(q, pos) - score(q, neg))) + lambda * |emb|^2
```

Hyperparams initiaux (à valider via revue SOTA) :
- `emb_dim = 1024` (égal à BGE-M3 pour préserver init)
- `n_layers = 3` (canonique LightGCN)
- `lr = 1e-3`, `batch_size = 2048`, `epochs = 100` (early stop)
- `lambda_reg = 1e-4`
- `negative_sampling = 1` négatif par positif, uniforme dans pool

### Étape 2 — Train + monitoring
- Logger loss train/val par epoch
- Eval sur val tous les 5 epochs sur **panel complet** (M1, MRR, NDCG @10 strict + étendu)
- Sauvegarder best checkpoint sur NDCG@10 étendu val
- Train en background si possible (use `run_in_background` ou tmux)

### Étape 3 — Inference + insertion tableau
- Sur cohorte test : pour chaque question, ranking articles ET ranking JP via produit scalaire
- Sauvegarder rankings dans `eval_lightgcn_rankings.parquet` (compat avec script M3 LLM-judge)
- Insérer ligne dans le grand tableau global

## Session 4 (optionnelle) — Variantes (4-6h)

Selon résultats LightGCN :
- Si bon : tenter **R-GCN inductif** (types d'arêtes Art↔JP différents)
- Si excellent : **HGT** (attention hétérogène, plus de capacité)
- Si médiocre : analyser pourquoi (init embeddings ? n_layers ? split ?) avant de scaler

## Fichiers clés

| Path | Rôle |
|---|---|
| `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/graph_v5.npz` | Graphe biparti existant (chargé via `config.GRAPH_NPZ`) |
| `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/articles_embeddings.npy` | Embeddings BGE-M3 articles (87821 × 1024) |
| `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/jp_synthese_embeddings.npy` | Embeddings BGE-M3 JP synthèses |
| `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/questions_977_emb.npy` | Embeddings BGE-M3 questions cohorte 971 |
| `05-Technique/benchmark/etape1_embedding_pur/etape1/config.py` | Config paths centralisé |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/20_ppr_naive.py` | Référence pour structure éval graphe |
| `02-Etat-de-l-art/gnn/LightGCN-2020.md` | Fiche papier (à créer) |
| **À créer** : `30_build_pyg_graph.py`, `31_lightgcn_train.py`, `32_lightgcn_eval.py` | Pipeline LightGCN |

## Pièges connus

- **Initialisation embeddings BGE-M3** : il faut **freeze** ou **fine-tune** les embeddings init ? Le papier LightGCN init aléatoire. On va innover en initialisant par BGE-M3 → choix non-trivial. Recommandation : fine-tune avec `lr` plus faible que les autres params, ou freeze sur les premières epochs.
- **Coût mémoire** : 87k articles + 118k JP + 971 questions × 1024-dim ≈ 800MB embeddings. OK GPU 16GB.
- **BPR loss vs autres** : LightGCN canonique = BPR. Alternatives : CCL, InfoNCE. Garder BPR au début pour fidélité.
- **Split leak** : si une question apparaît dans train ET test, leak. Vérifier que la cohorte 971 n'est **jamais** vue à l'entraînement (cf. session 2 étape 3).
- **Évaluation cohérente** : utiliser le même `metrics.py` que pour les baselines cosine/PPR. Pas de métrique custom LightGCN-specific.

## Critères de succès global (sur les 3-4 sessions)

- [ ] Lecture papier LightGCN + fichier Mattermost faite, fiches écrites
- [ ] Graphe converti en PyG HeteroData, splits validés sans leak
- [ ] Modèle LightGCN train sur GPU en convergence < 100 epochs
- [ ] Évaluation sur panel complet, ligne insérée dans le grand tableau
- [ ] Comparaison avec PPR : LightGCN bat sur ≥ 1 régime (strict ou étendu, art ou JP)
- [ ] Décision sur variantes (R-GCN, HGT) prise pour la suite

## Pour relancer ce travail

Coller en début de session Claude Code :
```
Reprends le handoff `01-Projet/handoffs/handoff-LightGCN.md` à partir
de la Session N (à préciser). Vérifie le statut des pré-requis (#32, #33,
Chantier 1) avant de démarrer. Demande confirmation à l'utilisateur
avant de modifier l'env Python ou de lancer un train cluster.
```

## Lien avec autres handoffs

- **Dépend de** : `handoff-M3-LLM-judge.md` (pour avoir M3 dans le tableau global) — pas bloquant pour le train LightGCN lui-même.
- **Alimente** : grand tableau global du Chantier 2.
