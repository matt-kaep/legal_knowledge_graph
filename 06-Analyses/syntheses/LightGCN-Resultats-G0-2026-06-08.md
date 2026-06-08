---
tags: [fiche, gnn, resultats, synthese]
type: synthese
created: 2026-06-08
modified: 2026-06-08
pertinence: haute
sujet: Résultats LightGCN (design A) sur graphe G0 — point Johnny
---

# LightGCN sur G0 — résultats (point Johnny, 2026-06-08)

> [!success] TL;DR
> LightGCN (design A : propagation sur le graphe de citations Art↔JP, questions = users BGE-M3, scoring **cosinus**) **bat PPR** sur articles-strict ET JP, **égale** le champion strict B3-e (gagne même NDCG/MRR), et devient le **champion JP** toutes méthodes confondues. Critère du handoff (« battre PPR sur ≥1 régime ») **atteint sur 2 régimes**. Sur le graphe **G0 brut** — avant tout nettoyage.

## 1. La décomposition qui raconte l'histoire (M1 strict, articles)

```
texte seul (cosine BGE-M3)            0,459
  + propagation graphe (K=2, figé)    0,653   (+0,194  ← apport du GRAPHE)
  + apprentissage (BPR cosinus K=2)   0,705   (+0,052  ← apport de l'APPRENTISSAGE)
```

Chaque étage ajoute. C'est l'argument central : **le graphe de citations ajoute massivement au sémantique pur, et l'apprentissage ajoute encore par-dessus.** L'ablation K=0 (= cosine) isole proprement chaque contribution.

## 2. Tableau global (cohorte 971, K=10, même pool/métriques pour tous)

### Articles
| méthode | M1_s | Hit_s | NDCG_s | M1_e | Hit_e | NDCG_e |
|---|---|---|---|---|---|---|
| B2-a (cosine articles) | 0,459 | 0,503 | 0,325 | 0,185 | 0,639 | 0,195 |
| B3-e (JP→Art via graphe) | **0,711** | **0,745** | 0,518 | 0,399 | 0,897 | 0,400 |
| PPR row α=0,85 | 0,601 | 0,647 | 0,334 | 0,421 | 0,926 | 0,435 |
| PPR row α=0,95 | 0,571 | 0,616 | 0,246 | **0,440** | 0,925 | **0,413** |
| LightGCN K=2 (BGE propagé) | 0,653 | 0,687 | 0,472 | 0,296 | 0,819 | 0,317 |
| **LightGCN K=2 entraîné** | 0,705 | 0,739 | **0,533** | 0,324 | 0,861 | 0,356 |

### JP
| méthode | M1 | Hit | NDCG |
|---|---|---|---|
| B3-a (cosine JP) | 0,408 | 0,427 | 0,304 |
| B4-e (RRF k_in=20) | 0,416 | 0,436 | 0,241 |
| PPR row α=0,95 | 0,407 | 0,426 | 0,236 |
| **LightGCN K=2 (BGE propagé)** | **0,437** | **0,457** | **0,318** |
| LightGCN K=2 entraîné | 0,426 | 0,446 | 0,311 |

**Lecture par régime** :
- **Articles strict** : LightGCN entraîné (0,705) ≈ champion B3-e (0,711) — et **gagne le NDCG** (0,533 vs 0,518) et le MRR. **Bat PPR** largement (NDCG 0,533 vs 0,334).
- **Articles étendu** : **PPR garde l'avantage** (M1 0,440, sa diffusion α=0,95 ratisse large). LightGCN plus précis mais moins exhaustif.
- **JP** : **LightGCN est le nouveau champion** (untrained 0,437 > B4-e 0,416 > tous).

## 3. Points de méthode (pour ne pas se faire piéger)

1. **Scoring cosinus obligatoire.** En produit scalaire, la propagation gonfle la norme des hubs → biais de popularité qui effondre strict+JP (M1 0,02). La L2-normalisation retire ce biais. *(C'est aussi pourquoi PPR-sym, droppé comme « ≡ cosine », se comporte ici très différemment : on lisse tout le champ d'embeddings, pas un seed de 10 items.)*
2. **Over-smoothing** visible : K0 < K1 < **K2** > K3. K=2 optimal, conforme à la théorie LightGCN.
3. **Apprentissage** : BPR **cosinus + température** (τ=0,1) + **ancrage** `λ‖E0−BGE‖²` anti-drift. Sans ces deux corrections, l'entraînement naïf (produit scalaire, 840 paires) **drift** et s'effondre. Avec, il **aide** (+0,052 strict).

## 4. Limites honnêtes

- **Données G0 brutes** : 36 % d'articles résolus, 48 % de nœuds morts, graphe binaire (cf. [[ADR-001-Versionnage-Graphe-G0-Vn]]). Le nettoyage G0→Vn est de l'**upside non exploité**.
- **840 paires d'entraînement** seulement (732 questions hors cohorte). Petit → l'apprentissage est de la calibration, pas un gros gain.
- **JP non supervisé** : l'entraînement (positifs = articles) érode légèrement le JP (0,437→0,426). Injecter du signal JP (régime D de l'ADR) = piste.
- **Étendu** : PPR reste devant. La diffusion large convient mieux au GT étendu.

## 5. Suites proposées

1. **Régime D (augmentation)** : injecter les questions cohorte JP+article au train → débloquer la supervision JP.
2. **G0 → V1** : retirer les 41 824 nœuds morts, re-mesurer (le programme ADR-001).
3. **Variante C** (questions dans le graphe) si on veut pousser l'étendu.

## Sources
- [[LightGCN-2020]] · [[Johnny-LightGCN-Solution-Notebook]] · [[sota-gnn-reco-2026]] · [[ADR-001-Versionnage-Graphe-G0-Vn]]
- Code : `scripts/31_lightgcn.py` · tableau : `scripts/24_build_global_table.py`
