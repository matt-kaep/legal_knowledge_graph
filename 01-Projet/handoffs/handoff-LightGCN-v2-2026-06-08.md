---
type: handoff
sujet: LightGCN design A — résultats sur G0 + suite (v2, remplace le plan initial)
date_creation: 2026-06-08
remplace: handoff-LightGCN.md (plan initial, sessions 1-4)
tasks_liees: [#31, #32, #33]
statut: Sessions 1-3 FAITES — LightGCN opérationnel, bat PPR sur 2 régimes
---

# Handoff v2 — LightGCN : état + suite après résultats sur G0

> [!success] Résumé exécutif
> Le prototype LightGCN est **fait et fonctionne**. Sur le graphe **G0** (brut), design A (propagation Art↔JP, questions = users BGE-M3, scoring cosinus), il **bat PPR** (la baseline graphe non-apprise) sur **articles-strict** ET **JP**, **égale** le champion strict B3-e, et devient le **champion JP**. Critère du handoff initial (« battre PPR sur ≥1 régime ») **atteint sur 2 régimes**. Tout est sur la branche `etape1-embedding-pur`.

---

## 1. Où on en est (état au 2026-06-08)

| Phase (handoff initial) | Statut |
|---|---|
| Session 1 — lecture + SOTA | ✅ fiches LightGCN, mémo SOTA, fiche notebook Johnny (#32 levé) |
| Pré-requis Chantier 1 (metrics.py) | ✅ déjà fait (Week-10), réutilisé |
| Session 2 — setup data | ✅ fait (pas de PyG : PyTorch pur) |
| Session 3 — implémentation + train | ✅ script 31 opérationnel, entraînement débloqué |
| Insertion grand tableau global | ✅ script 24 intègre LightGCN |
| Session 4 — variantes | ⬜ à faire (voir §6) |

---

## 2. Le résultat (cohorte 971, K=10, même pool/métriques que toutes les baselines)

**Décomposition (M1 strict, articles)** : texte seul `0,459` → +propagation graphe `0,653` (+0,194) → +apprentissage `0,705` (+0,052). Chaque étage ajoute.

**Articles**
| méthode | M1_s | NDCG_s | M1_e | NDCG_e |
|---|---|---|---|---|
| B2-a cosine | 0,459 | 0,325 | 0,185 | 0,195 |
| B3-e (JP→Art graphe) | **0,711** | 0,518 | 0,399 | 0,400 |
| PPR row α=0,85 | 0,601 | 0,334 | 0,421 | 0,435 |
| PPR row α=0,95 | 0,571 | 0,246 | **0,440** | **0,413** |
| LightGCN K=2 (BGE propagé) | 0,653 | 0,472 | 0,296 | 0,317 |
| **LightGCN K=2 entraîné** | 0,705 | **0,533** | 0,324 | 0,356 |

**JP**
| méthode | M1 | NDCG |
|---|---|---|
| B4-e RRF | 0,416 | 0,241 |
| PPR row α=0,95 | 0,407 | 0,236 |
| **LightGCN K=2 (BGE propagé)** | **0,437** | **0,318** |
| LightGCN K=2 entraîné | 0,426 | 0,311 |

**Lecture** : bat **PPR** sur strict + JP ; **égale** B3-e sur strict (gagne NDCG/MRR) ; **champion JP**. PPR garde l'**étendu**. Tradeoff entraînement : échange un peu de JP (0,437→0,426, pas de supervision JP) contre de la précision articles (0,653→0,705).

> [!warning] Fiabilité des claims
> - Victoire **vs PPR** (untrained K=2) = **déterministe** (aucun entraînement) → imprenable.
> - Gain **entraînement** (+0,052) = **1 seed, hyperparams non tunés** → à valider multi-seed.

---

## 3. Décisions d'architecture (design A, actées)

- **Design A** (vs B/C, cf. [[Johnny-LightGCN-Solution-Notebook]]) : graphe de propagation = **citations Art↔JP** (642k arêtes) ; les **questions ne sont PAS des nœuds** (users BGE-M3 figés, scorés de l'extérieur). Exploite notre actif unique (le graphe de citations).
- **Scoring COSINUS obligatoire** : en produit scalaire, la propagation gonfle la norme des hubs → biais de popularité qui effondre strict+JP. La L2-normalisation le retire. *(C'est pourquoi PPR-sym, droppé comme « ≡cosine », diffère ici : on lisse tout le champ, pas un seed de 10.)*
- **K=2** : over-smoothing visible K0<K1<K2>K3.
- **Entraînement** : BPR **cosinus** (τ=0,1) + **ancrage** `λ‖E0−BGE‖²` (λ=1,0, weight_decay=0). Sans ces 2 corrections, l'entraînement naïf drift et s'effondre.
- **Pool de ranking** = embeddés ∩ graphe (31 357 art, 116 755 JP), **identique aux baselines** (comparaison apples-to-apples, validée : cosine_raw == B2-a == 0,459 et == B3-a JP == 0,408).
- **Split anti-leak** : train = 708 questions doctrine_qgen **hors cohorte** (840 paires positives, GT étendu) ; cohorte 971 jamais vue (assert dans le code).

---

## 4. Fichiers clés

| Path (relatif à la racine du vault) | Rôle |
|---|---|
| `05-Technique/benchmark/etape1_embedding_pur/scripts/31_lightgcn.py` | **LE script** : modèle, propagation, BPR cosinus+ancrage, éval, 5 variantes |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/24_build_global_table.py` | Tableau global (intègre untrained_K2 + trained_K2) |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/20_ppr_naive.py` | Réutilisé par 31 (cohorte, graphe, pools, ranking) |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/metrics.py` | Panel M1/M2/Hit/MRR/NDCG (Chantier 1) |
| `06-Analyses/syntheses/LightGCN-Resultats-G0-2026-06-08.md` | **Synthèse Johnny** (slide-ready) |
| `01-Projet/decisions/ADR-001-Versionnage-Graphe-G0-Vn.md` | Stratégie G0→Vn + régimes supervision |
| `02-Etat-de-l-art/gnn/LightGCN-2020.md` | Fiche papier |
| `02-Etat-de-l-art/recommandation/Johnny-LightGCN-Solution-Notebook.md` | Fiche notebook Johnny (#32) |
| `06-Analyses/syntheses/sota-gnn-reco-2026.md` | Mémo SOTA GNN |
| Sorties (gitignorées) : `data/global_bench/lightgcn_eval.csv`, `lightgcn_summary.json`, `global_table.md` | Résultats |

---

## 5. Reproduire / relancer

Env : `.venv` du checkout principal (torch 2.12 + MPS, **PyG inutile**). Données + `.venv` vivent **uniquement dans le checkout principal** (gitignorés).

```bash
# Depuis la racine du vault (checkout principal) :
PY=05-Technique/benchmark/etape1_embedding_pur/.venv/bin/python
S=05-Technique/benchmark/etape1_embedding_pur/scripts

$PY -u $S/31_lightgcn.py              # diagnostic + entraînement K=2 (~5 min CPU)
NOTRAIN=1 $PY $S/31_lightgcn.py       # diagnostic seul (untrained), ~70s
SEED=1   $PY $S/31_lightgcn.py        # autre seed (validation multi-seed)
TRAIN_K=1 $PY $S/31_lightgcn.py       # entraîner à un autre K
$PY $S/24_build_global_table.py       # reconstruit le grand tableau
```

⚠️ Les runs écrivent `lightgcn_eval.csv` (clobber) → pour le multi-seed, sauvegarder/suffixer les sorties, et finir par un run SEED=42 pour restaurer le canonique.

---

## 6. Prochaines étapes (par priorité)

1. **Valider l'entraînement multi-seed** (3 seeds, mean±range) + ablation init (zéro vs aléatoire) — durcit le +0,052 avant publication. `SEED=n` prêt.
2. **Régime D (supervision JP)** : injecter les questions cohorte JP+article au train (k-fold ou +50 %, table séparée, cf. [[ADR-001-Versionnage-Graphe-G0-Vn]]) → débloquer l'apprentissage JP (actuellement érodé).
3. **G0 → V1** : retirer les 41 824 nœuds morts du pool, re-mesurer (programme ADR-001). Quantifie l'apport du nettoyage.
4. **Combler l'étendu** : tester la **variante C** (questions dans le graphe) ou un α-mix style PPR ; PPR garde l'avantage en rappel large.
5. **Tuning** : τ, λ, lr, epochs (grille légère). K autour de 2.
6. **Slide Johnny** : la synthèse §4 est slide-ready en **markdown** — la convertir au format **deck du projet** (`01-Projet/presentations/Week-N-*.tex` beamer / HTML) si présentation formelle.

---

## 7. Pièges connus (appris cette session)

- **Scoring** : toujours cosinus (cf. §3). Le produit scalaire donne M1≈0,02 (collapse hubs).
- **Entraînement naïf** (produit scalaire, sans ancrage) : drift sur 840 paires → s'effondre. Toujours cosinus + ancrage.
- **Git/worktree** : l'infra + données vivent dans le **checkout principal** (`etape1-embedding-pur`) ; `31_lightgcn.py` hardcode `REPO` vers ce chemin → on peut l'éditer dans un worktree mais il lit/écrit le principal. Données gitignorées **non** présentes dans les worktrees frais.
- **Buffering** : piper un run via `tee`/`grep` bufferise stdout ; utiliser `python -u` pour voir la progression.
- **Sanity check obligatoire** : toute nouvelle méthode doit reproduire une baseline connue (ici cosine_raw == B2-a) avant d'interpréter.

---

## 8. État Git

Tout est mergé sur **`etape1-embedding-pur`** (checkout principal). Worktree `lightgcn-session1` synchronisé. Commits clés : fiches Session 1, ADR-001, script 31 (design A), entraînement cosinus, tableau global, cette synthèse.

## Lien avec autres handoffs
- Remplace `handoff-LightGCN.md` (plan initial).
- Alimente le grand tableau global du Chantier 2.
- Voisin : `handoff-M3-LLM-judge.md` (M3 dans le tableau).
