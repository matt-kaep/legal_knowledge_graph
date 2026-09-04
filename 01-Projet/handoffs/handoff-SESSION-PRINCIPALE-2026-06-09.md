---
type: handoff
sujet: Session principale — état du benchmark Étape 1 (Chantiers 1-4 + M3), branche etape1-embedding-pur
date_creation: 2026-06-09
remplace: —
statut: Chantiers 1, 2, 4 FAITS ; M3 en gate Johnny ; Chantier 3 à faire
---

# Handoff — Session principale (benchmark Étape 1)

> [!abstract] Pour reprendre
> Tout le travail vit sur la branche **`etape1-embedding-pur`** (checkout principal). Le cadre opérationnel = les **4 chantiers Week-10 + M3** (cf. `01-Projet/presentations/Week-10-2026-06-09-Presentation-Superviseur.tex`). Données + `.venv` sont dans le checkout principal (gitignorés). Point Johnny Week-10 = **9 juin midi**.

---

## 1. État des chantiers (Week-10)

| Chantier | Statut | Détail |
|---|---|---|
| **1 — Panel métriques** | ✅ FAIT | `scripts/metrics.py` : M1, M2, Hit@K, MRR@K, NDCG@K + `panel_strict_ext`. Conventions actées (MRR cappé K, NDCG binaire, GT vide→NaN). |
| **2 — Grand tableau global** | ✅ FAIT (+LightGCN) | `scripts/24_build_global_table.py` lit 18 (B*) + 20 (PPR) + 31 (LightGCN). Sorties `global_table_{articles,jp}.csv` + `.md`. |
| **3 — Enrichir le benchmark** | ⬜ À FAIRE | GT multi-niveau (central/support/contextuel) → NDCG multi-niveau ; sweep K_in (#27, figé à 10). |
| **4 — LightGCN** | ✅ FAIT (G0) | Bat PPR sur 2 régimes, validé multi-seed. → `handoff-LightGCN-v2-2026-06-08.md`. |
| **M3 — LLM-as-judge** | ⏳ GATE JOHNNY | Plomberie prête (mock OK), vLLM local GPU-only, 46,6k paires réelles, **dénominateur en attente Johnny** avant run cluster. → `handoff-M3-LLM-judge.md`. |

---

## 2. Le grand tableau global — état actuel (cohorte 971, K=10)

**Articles** (champions en gras)
| méthode | M1 strict | NDCG strict | M1 étendu | NDCG étendu |
|---|---|---|---|---|
| B3-e (JP→Art graphe) | **0,711** | 0,518 | 0,399 | 0,400 |
| LightGCN K=2 entraîné | 0,704 | **0,533** | 0,326 | 0,356 |
| LightGCN K=2 (BGE propagé) | 0,653 | 0,472 | 0,296 | 0,317 |
| PPR row α=0,85 | 0,601 | 0,334 | **0,421** | **0,435** |
| B2-a (cosine) | 0,459 | 0,325 | 0,185 | 0,195 |

**JP**
| méthode | M1 | NDCG |
|---|---|---|
| **LightGCN K=2 (BGE propagé)** | **0,437** | **0,318** |
| B4-e (RRF k_in=20) | 0,416 | 0,241 |
| B3-a (cosine JP) | 0,408 | 0,304 |
| PPR row α=0,95 | 0,407 | 0,236 |

**Lectures clés** (cf. journal 2026-06-08) :
- **Strict articles** : B3-e champion, **LightGCN entraîné l'égale** (gagne NDCG/MRR).
- **Étendu articles** : **PPR α=0,85** champion (4/5 métriques) — l'avantage α=0,95 sur le seul M1 était trompeur.
- **JP** : **LightGCN nouveau champion** ; auparavant PPR JP était saturée (= cosine, « le graphe n'apporte rien côté JP ») → **LightGCN renverse ce constat**.
- Tension non tranchée : **B4-e RRF vs B3-a cosine** côté JP (couverture vs ordre).

---

## 3. Résultat phare de la session : LightGCN (Chantier 4)

Sur le graphe **G0 brut**, design A (propagation Art↔JP, questions = users BGE-M3 figés, scoring **cosinus**) :
- **Décomposition** (M1 strict) : texte 0,459 → +graphe 0,653 → +appris **0,704** (validé 3 seeds, ±0,007).
- **Bat PPR** sur articles-strict ET JP ; **égale** B3-e ; **champion JP**.
- Détails complets, design, pièges, reproduction → **`handoff-LightGCN-v2-2026-06-08.md`** + synthèse `06-Analyses/syntheses/LightGCN-Resultats-G0-2026-06-08.md` + deck `01-Projet/presentations/LightGCN-Resultats-G0-2026-06-08.tex`.

---

## 4. État Git / consolidation (IMPORTANT)

Le projet était **fragmenté** sur plusieurs branches/worktrees ; **tout est désormais sur `etape1-embedding-pur`** :
- Infra Week-10 (metrics.py, scripts 18/20/24, handoffs) — ex-branche `etat-lieux-johnny-2026-05-28`, mergée.
- Travail LightGCN (scripts 31, fiches, ADR-001, synthèse, deck) — ex-worktree `lightgcn-session1`, mergé.
- MAJ Week-10 unifiée (M3 + LightGCN) — committée sur etape1.

⚠️ **Données + `.venv` vivent UNIQUEMENT dans le checkout principal** (gitignorés, absents des worktrees frais). Les scripts qui hardcodent `REPO` (ex. 20, 31) lisent/écrivent ce checkout.

---

## 5. Point Johnny Week-10 — décisions en attente

1. **M3 dénominateur** : définition à valider avant run cluster (gate).
2. **Arbitrage JP** : B4-e RRF (couverture) vs B3-a cosine (ordre) — méthode de référence JP.
3. **Sweep K_in** PPR (#27, figé à 10) : à lancer ?
4. **Chantier 3** : niveaux de GT (central/support/contextuel) → NDCG multi-niveau.
5. **LightGCN suite** : régime D (supervision JP), G0→V1 — valider le principe G0→Vn (ADR-001).

---

## 6. Prochaines étapes (par chantier)

- **Chantier 3** : enrichir GT (multi-niveau), NDCG multi-niveau, sweep K_in (#27).
- **M3** : après gate Johnny → run cluster (vLLM), insérer M3 dans le tableau global.
- **LightGCN** : ablation init (zéro vs aléatoire), régime D, **G0→V1** (retirer 41 824 nœuds morts), tuning τ/λ/lr.
- **Tableau global** : ajouter colonnes M3 quand dispo.

---

## 7. Fichiers clés (tous sur `etape1-embedding-pur`)

| Path | Rôle |
|---|---|
| `05-Technique/benchmark/etape1_embedding_pur/scripts/metrics.py` | Panel métriques (Chantier 1) |
| `…/scripts/18_eval_m1_m2.py` · `20_ppr_naive.py` · `31_lightgcn.py` | Baselines B* · PPR · LightGCN |
| `…/scripts/23_eval_m3_llm_judge.py` | M3 (gate Johnny) |
| `…/scripts/24_build_global_table.py` | Grand tableau global |
| `01-Projet/decisions/ADR-001-Versionnage-Graphe-G0-Vn.md` | Stratégie G0→Vn |
| `06-Analyses/syntheses/{LightGCN-Resultats-G0,sota-gnn-reco}-*.md` | Synthèses |
| `01-Projet/handoffs/handoff-{LightGCN-v2,M3-LLM-judge,M3-collecte-agregation}.md` | Sous-handoffs |
| `01-Projet/presentations/Week-10-…tex` · `LightGCN-Resultats-G0-…tex` | Decks Johnny |

---

## 8. Conventions / pièges (appris)

- **Sanity check obligatoire** : toute nouvelle méthode reproduit une baseline connue avant interprétation (ex. LightGCN cosine_raw == B2-a == 0,459).
- **Scoring cosinus** pour les méthodes à embeddings propagés (sinon biais de norme/hub).
- **Anti-leak** : la cohorte 971 n'est jamais vue à l'entraînement (assert dans le code).
- **Env** : ne pas polluer le `.venv` ni l'env `--user` cluster (cf. mémoire `cluster-user-env-fragile`). PyG **inutile** (PyTorch pur).
- **pdflatex absent localement** : compiler les decks côté projet.

## Lien avec autres handoffs
- `handoff-LightGCN-v2-2026-06-08.md` (détail Chantier 4)
- `handoff-M3-LLM-judge.md` + `handoff-M3-collecte-agregation.md` (M3)
- Remplace de facto le pilotage dispersé : ce fichier = point d'entrée session principale.
