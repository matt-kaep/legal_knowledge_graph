---
tags: [article, gnn, recommandation]
categorie: "GNN"
titre_complet: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
auteurs: "Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang"
annee: 2020
type: "Conférence (SIGIR)"
venue: "SIGIR 2020"
url: "https://arxiv.org/abs/2002.02126"
doi: "10.1145/3397271.3401063"
pdf_local: ""
status: "lu"
pertinence: "haute"
created: 2026-06-08
modified: 2026-06-08
---

# LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

> [!info] Metadonnees
> **Auteurs** : He, Deng, Wang, Li, Zhang, Wang
> **Annee** : 2020 | **Venue** : SIGIR 2020
> **Type** : Article de conférence (baseline reco standard, très cité)
> **URL** : [arxiv.org/abs/2002.02126](https://arxiv.org/abs/2002.02126)

> [!note] Provenance
> Fiche rédigée **de mémoire** (sans relecture du PDF). Les **hyperparamètres canoniques** (K=3, dim 64, lr 1e-3, λ 1e-4, BPR) sont fiables et suffisent à configurer le train. À **vérifier sur le PDF avant citation dans le mémoire** : le DOI, le chiffre « +16 % » et les valeurs absolues de Recall@20/NDCG@20.

## Resume

LightGCN part d'un constat empirique : NGCF (Neural Graph Collaborative Filtering) hérite de GCN deux opérations qui sont **inutiles, voire nuisibles**, pour le filtrage collaboratif — la **transformation de features** (matrices de poids `W`) et l'**activation non-linéaire**. En collaborative filtering, les nœuds n'ont pas de features sémantiques d'entrée (juste des embeddings d'ID appris) : ces deux briques n'apportent rien sauf de la difficulté d'optimisation. LightGCN les **supprime** et ne garde que l'agrégation de voisinage normalisée. Le modèle résultant est plus simple, plus rapide, et bat NGCF de ~16 % en moyenne sur Recall@20 / NDCG@20. Conceptuellement, c'est une **propagation linéaire pondérée** très proche d'APPNP / Personalized PageRank — d'où son intérêt direct comme « PPR apprise » pour notre projet.

## Contributions principales

1. **Ablation diagnostique de NGCF** : montre que retirer la transformation de features *et* la non-linéarité améliore les performances. Les deux opérations ne se justifient que quand les nœuds ont des features sémantiques en entrée — pas le cas en CF pur.
2. **Light Graph Convolution (LGC)** : une couche de propagation réduite à sa plus simple expression — agrégation normalisée symétrique des voisins, **sans self-loop, sans `W`, sans activation**.
3. **Layer combination** : l'embedding final est la **moyenne** (ou somme pondérée) des embeddings de toutes les couches `0..K`. Cette combinaison joue le rôle de self-connection et atténue le sur-lissage (over-smoothing).
4. **Analyse théorique** : connexion explicite à SGC (Simplifying Graph Convolution) et **APPNP** (propagation de type Personalized PageRank), et analyse au second ordre montrant comment LightGCN lisse le signal collaboratif.

## Methodologie

### Donnees
- Source : 3 datasets reco publics — **Gowalla** (check-ins), **Yelp2018**, **Amazon-Book**.
- Volume : graphes user–item bipartites, dizaines de milliers d'users/items, ~10⁶ interactions.
- Langue : N/A (IDs, pas de texte).

### Architecture / Pipeline

**Seuls paramètres entraînables** : les embeddings d'ID de la couche 0, `E^(0)` (users + items). Aucune matrice de transformation.

**Light Graph Convolution (propagation d'une couche)** :
```
e_u^(k+1) = Σ_{i ∈ N_u}  1 / ( sqrt(|N_u|) · sqrt(|N_i|) ) · e_i^(k)
e_i^(k+1) = Σ_{u ∈ N_i}  1 / ( sqrt(|N_i|) · sqrt(|N_u|) ) · e_u^(k)
```
- Normalisation **symétrique** `D^{-1/2} A D^{-1/2}` (le « DAD » du handoff).
- **PAS de self-connection** (le terme `e_u^(k)` lui-même n'est pas ré-injecté dans la somme — il revient via la layer combination).
- **PAS** de `W` (feature transform), **PAS** de σ (non-linéarité).

**Forme matricielle** : `E^(k+1) = (D^{-1/2} A D^{-1/2}) E^(k)`, où `A` est la matrice d'adjacence du graphe biparti user–item.

**Layer combination (lecture finale)** :
```
e_u = Σ_{k=0}^{K} α_k · e_u^(k)          avec α_k = 1/(K+1)  (moyenne uniforme par défaut)
```

**Prédiction** : score = produit scalaire `ŷ_ui = e_u^T · e_i`.

**Perte — BPR (Bayesian Personalized Ranking)** :
```
L = - Σ_{(u,i,j)} ln σ( ŷ_ui − ŷ_uj )  +  λ · ‖E^(0)‖²
```
Pour chaque positif `(u,i)` observé, on échantillonne un négatif `j` non observé ; on pousse le score du positif au-dessus du négatif. **Seul `E^(0)` est régularisé** (L2), pas les embeddings propagés. Pas de dropout nécessaire.

### Hyperparamètres canoniques (à reprendre comme défaut)

| Hyperparam | Valeur papier | Note pour notre setup |
|---|---|---|
| Dim embedding | **64** | 1024 chez nous pour préserver l'init BGE-M3 |
| Nb couches `K` | **3** (parfois 4) | testé 1–4 ; gain marginal au-delà de 3 |
| Optimiseur | **Adam** | |
| Learning rate | **1e-3** | |
| λ (L2 reg) | **1e-4** | recherche sur [1e-6, 1e-2] |
| Batch size | **2048** | |
| `α_k` | **1/(K+1)** uniforme | la version apprenable n'aide pas |
| Négatifs | **1 par positif**, uniforme | |
| Init | normal / Xavier | notre innovation : init BGE-M3 |

## Resultats cles

| Metrique | LightGCN vs NGCF | Note |
|---|---|---|
| Recall@20 (moyenne 3 datasets) | **≈ +16 %** | gain net et reproductible |
| NDCG@20 | gain comparable | |

> Les chiffres absolus dépendent du dataset (Gowalla / Yelp2018 / Amazon-Book). Le point retenu : **LightGCN > NGCF > Mult-VAE / GRMF** de façon nette, avec **moins de paramètres** et un entraînement plus rapide.

## Points forts

- **Simplicité + performance** : moins de paramètres que NGCF, meilleur résultat, train plus rapide.
- **Pas de features requises** : fonctionne sur IDs seuls → baseline universelle.
- **Robustesse à l'over-smoothing** via la layer combination (≠ GCN profond qui s'effondre).
- **Interprétabilité** : propagation linéaire analysable, parenté claire avec PPR/APPNP.

## Limites

- **Transductif** : suppose users/items fixes vus à l'entraînement. Pas inductif nativement (un nouvel item sans interaction n'a pas d'embedding) → cf. CaseLink pour la variante inductive.
- **Pas de features sémantiques** par design : ne sait pas exploiter du contenu (texte, embeddings pré-entraînés). **C'est précisément le point où notre projet innove** en initialisant par BGE-M3.
- **Homogène biparti** : un seul type d'arête user–item. Les relations typées (Art↔JP vs Art↔Art) demandent R-GCN / HGT.
- BPR = ranking par paires ; alternatives (InfoNCE, CCL) parfois meilleures mais hors scope initial.

## Liens avec mon projet

> [!important] Pertinence pour le KG juridique français
> LightGCN est la **transition naturelle depuis notre baseline PPR** (`20_ppr_naive.py`, α=0,95 row-norm). PPR propage un signal *non appris* sur le graphe de citations ; LightGCN propage des embeddings *appris* par la même mécanique (`D^{-1/2} A D^{-1/2}`). C'est donc une **PPR apprise** — Johnny l'a explicitement nommée Week-9 comme baseline GNN incontournable.

### Ce que je peux reutiliser
- **Mécanique de propagation** : on a déjà le graphe biparti `graph_penal.npz` en CSR (`D^{-1/2} A D^{-1/2}` = quelques lignes scipy/torch.sparse).
- **Hyperparams par défaut** : `K=3`, `lr=1e-3`, `λ=1e-4`, BPR, 1 négatif/positif.
- **Layer combination par moyenne** : trivial à implémenter, atténue l'over-smoothing « gratuitement ».

### Ce que je dois adapter
- **Init BGE-M3 au lieu d'aléatoire** : innovation non triviale (freeze vs fine-tune ; lr plus faible sur les embeddings init). Le papier init aléatoire → on exploite le signal sémantique pré-existant pour accélérer la convergence et secourir les nœuds faiblement connectés.
- **Question = user virtuel** : nos « users » sont des questions juridiques (cohorte 971), initialisées par leur embedding BGE-M3, jamais vues à l'entraînement (split sans leak).
- **Contrainte de couverture & cartographie des nœuds** (mesurée le 2026-06-08) : sur les **87 821 articles** du graphe, seuls **31 357** sont embeddés (texte résolu) **et 59 945 ne sont jamais cités** (degré 0). Croisement des deux axes :
  | | Embeddé | Non embeddé |
  |---|---|---|
  | **Cité** (degré ≥ 1) | 13 236 (signal max) | 14 640 (init aléatoire, **apprend via propagation**) |
  | **Jamais cité** (degré 0) | 18 121 (texte seul = cosine) | **41 824 doublement morts** (inertes) |
  → Correction d'une assertion antérieure : un nœud non-embeddé n'« apprend par propagation » **que s'il est cité** ; les 41 824 doublement morts restent figés aléatoires et ne sont jamais retrouvables. **Mitigation** : restreindre le pool de candidats aux ~46 000 articles vivants (cités ∪ embeddés). **Impact GT cohorte minuscule** : seulement 9 articles GT morts, 7 questions strictes 100 % perdues. Voir [[ADR-001-Versionnage-Graphe-G0-Vn]] (retrait des nœuds morts = V1).
- **Dim 1024** (vs 64 papier) pour préserver l'init BGE-M3 — coût mémoire ~800 Mo d'embeddings, OK GPU 16 Go.

## Connexions

### Articles lies
- [[Tang-2024-CaseLink-Inductive-Graph-Learning]] — variante **inductive** (généralise à de nouveaux cas), répond à la limite transductive de LightGCN
- [[Wang-2022-Legal-Judgment-Heterogeneous-Graphs]] — graphes **hétérogènes** juridiques (pendant R-GCN/HGT)
- [[Wendlinger-2025-Joint-Legal-Citation-Prediction]] — prédiction de citations légales par graphe

### Concepts lies
- [[PPR-Personalized-PageRank]] — LightGCN ≈ PPR apprise ; baseline directe à battre
- [[Embeddings-BGE-M3]] — source de l'initialisation sémantique

### Questions soulevees
- Init BGE-M3 : **freeze** les premières epochs puis fine-tune, ou fine-tune dès le début avec `lr` réduit ? ⚠️ **Interaction avec les 56 464 articles non-embeddés** : ces nœuds reçoivent un layer-0 aléatoire ; **freezer l'init les figerait aléatoires → irrécupérables**. La conclusion « non bloquant » ne tient donc que si ces nœuds sont **fine-tunés** (ou qu'on ne freeze jamais). À trancher en Session 3.
- Le gain LightGCN vs PPR vient-il de l'apprentissage ou juste de la dim 1024 ? (ablation : LightGCN K=0 = pur lookup BGE-M3 = notre cosine baseline)

## Citations cles

> « We argue that the two operations — feature transformation and nonlinear activation — contribute little to the effectiveness of NGCF. Even worse, they add to the difficulty of training. » (Abstract / §3)

> « The only trainable model parameters are the embeddings at the 0-th layer. » (§3.1)

## Notes personnelles

Le point le plus important pour nous : **l'ablation K=0**. Avec `K=0` (aucune propagation), LightGCN se réduit au produit scalaire des embeddings d'ID = exactement notre **baseline cosine BGE-M3**. Chaque couche ajoutée mesure donc précisément le gain apporté par la propagation sur le graphe de citations. C'est l'expérience qui isole proprement « ce que le graphe apporte au-dessus du sémantique pur » — argument central du mémoire. À instrumenter dès la première éval (M1/MRR/NDCG @ K=0,1,2,3).
