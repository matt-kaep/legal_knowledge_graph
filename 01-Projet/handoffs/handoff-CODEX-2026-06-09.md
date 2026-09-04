---
type: handoff
sujet: Handoff Codex — reprise complète projet Legal Knowledge Graph (Étape 1)
date_creation: 2026-06-09
destinataire: Codex (OpenAI) ou tout agent IA capable de Bash + Edit/Write
statut: point Johnny Week-10 réalisé (9 juin) ; reprise pour Week-11
duree_lecture_estimee: 15 min
---

# Handoff CODEX — Projet Legal Knowledge Graph (Étape 1)

> **TL;DR pour reprendre la session sans contexte préalable**
>
> Projet de recherche : construire un Knowledge Graph juridique français pour la recherche d'articles + jurisprudence à partir de questions. **Étape 1 (en cours)** : benchmark de méthodes de retrieval (cosine, graphe non-appris, PPR, LightGCN) sur 971 questions. Point superviseur ("Johnny") **Week-10 fait le 9 juin 2026**. Tout le travail est sur la branche **`etape1-embedding-pur`**. Pour comprendre l'état complet, lire dans cet ordre : (1) ce fichier ; (2) `01-Projet/presentations/Week-10-2026-06-09-Presentation-Superviseur.pdf` (25 pages, le deck de la réunion d'aujourd'hui) ; (3) `01-Projet/journal/2026-06-08.md` (résultats du jour) ; (4) `01-Projet/handoffs/handoff-LightGCN-v2-2026-06-08.md` et `handoff-M3-COMPLET.md` pour le détail des 2 gros chantiers.

---

## 0. Identité et environnement

- **Utilisateur** : Matthieu Kaeppelin, stagiaire FE Recherche, Telecom Paris.
- **Supervisor** : Johnny (Hector) — point hebdomadaire midi, mercredi typiquement.
- **OS** : macOS Darwin 25.2.0 (zsh).
- **Repo principal (source de vérité)** : `/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/`
- **Branche active** : `etape1-embedding-pur` (commits récents sur `origin/etape1-embedding-pur`).
- **Branche principale** : `main` (cible de merge éventuelle, pas mergée actuellement).
- **Outils essentiels** :
  - Python 3.12.3 (`/Users/matthieu.kaeppelin/.pyenv/versions/3.12.3/bin/python`)
  - `.venv` dédié : `05-Technique/benchmark/etape1_embedding_pur/.venv/` (PyTorch 2.12 + MPS, **PyG inutile**)
  - `pdflatex` : **PAS dans le PATH par défaut**, chemin absolu `/usr/local/texlive/2025/bin/universal-darwin/pdflatex`
  - `git`, `gh` disponibles
- **Mémoires utilisateur (à respecter strictement)** :
  - **Pas de copier-coller depuis le terminal** : ne JAMAIS donner à l'utilisateur des blocs de commandes à exécuter manuellement. Fournir un `.sh` exécutable ou exécuter via Bash directement.
  - **Env --user du cluster fragile** : ne JAMAIS faire `pip install` non-pinné sur le cluster Hector. Vérifier l'env existant d'abord.
  - **Vérifier sur preuve, pas supposition** : avant de conclure (« ça marche », « c'est fait »), test discriminant obligatoire (re-run un sanity check connu).
  - Responses en **français** avec **accents corrects** (toujours).

---

## 1. Contexte projet — le problème scientifique

### 1.1 Objectif final

Construire un système de recherche juridique : à partir d'une **question** (formulée par un avocat, un juriste, un justiciable), retrouver :
- les **articles de loi pertinents** (codes : pénal, procédure pénale, etc.)
- les **arrêts (jurisprudence)** qui appliquent ces articles

Vision long terme = un Knowledge Graph qui relie Articles ↔ JP ↔ Concepts juridiques ↔ Questions, exploité par des méthodes de reco (GNN) ou de retrieval (LLM+RAG).

### 1.2 Architecture du benchmark (Étape 1)

- **Cohorte d'éval** : 971 questions doctrine_qgen + 1 CRFPA (sur 977 nominal, 6 droppées car embeddings BGE-M3 manquants).
- **GT** (ground truth) :
  - GT articles **strict** : articles_attendus de la question (|GT| moy = 1,23 — quasi-singleton).
  - GT articles **étendu** : strict + articles cités par les JP attendues (|GT| moy = 7,39).
  - GT JP : pourvois CC résolus dans Judilibre (|GT| moy = 1,13).
- **Graphe biparti existant** : 118 112 JP × 87 821 articles, ~642k citations (graphe G0 brut).
- **Embeddings** : BGE-M3 (1024-dim, ctx 8k). Articles 31 357/87 821 embeddés, JP 116 755/118 112 embeddés (cf. mémoire `lightgcn-data-reality`).

### 1.3 Panel de méthodes testées

| Famille | Méthodes | Statut |
|---|---|---|
| **Cosine pur** | B2-a (articles), B3-a (JP) | testé |
| **Graphe non-appris** | B3-e (JP→Art via citations), B4-a/c/d/e/f (cross-modal union/intersection/RRF/citation-weighted) | testé |
| **PPR** | row-norm × α ∈ {0,5; 0,7; 0,85; 0,95} ; sweep complet (s × seed × α) = 48 configs | sweep complet fait |
| **LightGCN** | Design A : propagation BGE-M3 sur graphe citations, scoring cosinus, K=2 | fait + validé 3 seeds |
| **LLM seul / LLM+RAG** | Gemma 4 26B générant directement | non implémenté |

### 1.4 Panel de métriques (figé Week-10)

Module : `05-Technique/benchmark/etape1_embedding_pur/scripts/metrics.py`

| Métrique | Définition | Régime cible |
|---|---|---|
| **M1** | Recall@K = \|GT ∩ R[:K]\| / \|GT\| | multi-GT (étendu) |
| **M2** | rang moyen normalisé (custom, voir code) | multi-GT |
| **Hit@K** | 1 si ≥1 GT dans top-K sinon 0 | singleton |
| **MRR@K** | 1/rang du premier GT, cappé à K | singleton |
| **NDCG@K** | rel binaire {0,1}, DCG/IDCG | multi-GT |
| **M3** | LLM-as-judge (Gemma 4 26B), agrégation (2·#n2 + #n1)/(2K) | qualité hors-GT |

K=10 partout. Régime strict (|GT|=1) vs étendu (|GT|≈7) reportés séparément côté articles.

**Notation** : `s` (seed size) pour les hyperparamètres comme B3-e s=10, PPR s=5, etc. (anciennement `K_in`, changé pour ne pas confondre avec K=top-K).

---

## 2. État Git — histoire des worktrees (IMPORTANT)

Le projet a été **fragmenté** sur plusieurs branches/worktrees pendant Week-9 → Week-10, **tout est désormais consolidé sur `etape1-embedding-pur`**. Important à comprendre :

### 2.1 Worktrees créés et utilisés

| Worktree | Branche | Travail | Statut |
|---|---|---|---|
| (checkout principal) | `etape1-embedding-pur` | **source de vérité actuelle** | actif |
| `.claude/worktrees/etat-lieux-johnny-2026-05-28/` | `worktree-etat-lieux-johnny-2026-05-28` | infra Week-10 : metrics.py, scripts 18/20/24, présentation Week-10 unifiée (M3+LightGCN), handoffs | mergé dans etape1 |
| `.claude/worktrees/worktree-lightgcn-session1/` | `worktree-lightgcn-session1` | LightGCN design A + entraînement + multi-seed | mergé dans etape1 |

### 2.2 Commits récents (de plus récent au plus ancien)

```
15bd969 docs(slides): Week-10 alignée multi-seed LightGCN (3 seeds validés)  ← dernier
1779069 docs(handoff): handoff session principale (Chantiers 1-4 + M3)
ae0ff12 Merge branch 'worktree-lightgcn-session1' into etape1-embedding-pur
5cd9d53 docs(lightgcn): validation multi-seed (3 seeds) consolidée
c46da05 docs(slides): Week-10 unifiée — M3 LLM-judge + LightGCN intégrés
b3b4d2f docs(slides): deck beamer résultats LightGCN G0 (point Johnny)
4b1e21d docs(handoff): handoff LightGCN v2 + cadrage résultats + SEED configurable
c0958a5 feat(lightgcn): entraînement cosine-BPR + intégration tableau global
a212fee feat(lightgcn): script 31 design A — propagation BGE + cosine bat PPR
1edaeea docs(reco): fiche notebook LightGCN de Johnny (#32 levée)
82a2c51 Merge branch 'worktree-lightgcn-session1' into etape1-embedding-pur
c05ae11 Merge branch 'worktree-etat-lieux-johnny-2026-05-28' into etape1-embedding-pur
0771d22 fix(docs): Chantier 1 (metrics.py) FAIT, pas à implémenter
f7b00e1 docs(adr): ADR-001 versionnage graphe G0→Vn + régimes supervision
070a508 feat(eval): panel métriques complet + grand tableau global Week-10
```

### 2.3 IMPORTANT : où vivent les données

**Les données + `.venv` vivent UNIQUEMENT dans le checkout principal**, ne sont **PAS** dans les worktrees (gitignorées dans `.gitignore`). Les scripts hardcodent `REPO = Path("/Users/.../legal_knowledge_graph")` (chemin absolu) → ils lisent/écrivent du checkout principal **même si exécutés depuis un worktree**.

Conséquences pratiques :
- Pour **exécuter les scripts**, travailler dans le checkout principal (ou s'assurer qu'il existe).
- Pour **modifier le code**, peu importe (le worktree marche, on cp ou commit ensuite).
- Pour **modifier la présentation** : préférable de travailler directement dans le checkout principal pour éviter le cp manuel.

### 2.4 Recommandation Codex : travailler sur le checkout principal

Pour Codex, le plus simple est de **travailler directement sur `/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/`** sur la branche `etape1-embedding-pur`. Pas besoin de worktree — la session Claude qui a précédé a utilisé des worktrees pour isoler le travail mais c'était par convention, pas nécessité.

---

## 3. État des chantiers Week-10 (et au-delà)

### 3.1 Vue d'ensemble

| Chantier | Statut | Owner / handoff |
|---|---|---|
| **1 — Panel métriques** | ✅ FAIT | metrics.py + sanity check |
| **2 — Grand tableau global** | ✅ FAIT (+ LightGCN + M3) | 24_build_global_table.py |
| **3 — Enrichissement bench** | ⬜ À FAIRE | mis de côté (GT multi-niveau, reverse-eng décisions, CRFPA) |
| **4 — LightGCN** | ✅ FAIT (G0 brut) | handoff-LightGCN-v2-2026-06-08.md |
| **M3 — LLM-as-judge** | ✅ FAIT (run cluster) | handoff-M3-COMPLET.md |
| Sweep PPR (s × seed × α) | ✅ FAIT | ppr_kin_sweep_analysis.md |

### 3.2 Le grand tableau global — résultats actuels

**Côté ARTICLES** (champions en gras) :

| méthode | M1 strict | NDCG strict | M1 étendu | NDCG étendu | M3 |
|---|---|---|---|---|---|
| B2-a (cosine) | 0,459 | 0,325 | 0,185 | 0,195 | **0,325** |
| B3-e (JP→Art s=10) | **0,711** | 0,518 | 0,399 | 0,400 | 0,317 |
| PPR row α=0,85 (s=10 both) | 0,601 | 0,334 | 0,421 | 0,435 | 0,296 |
| PPR row α=0,95 (s=10 both) | 0,571 | 0,246 | 0,440 | 0,413 | 0,214 |
| LightGCN K=2 untrained | 0,653 | 0,472 | 0,296 | 0,317 | — |
| **LightGCN K=2 entraîné** | 0,704 ± 0,007 | **0,532 ± 0,003** | 0,324 | 0,356 | — |

**Côté JP** :

| méthode | M1 | NDCG | M3 |
|---|---|---|---|
| B3-a (cosine JP) | 0,408 | 0,304 | **0,672** |
| B4-e (RRF s=20) | 0,416 | 0,241 | 0,589 |
| PPR row α=0,95 | 0,407 | 0,236 | 0,632 |
| **LightGCN K=2 untrained** | **0,437** | **0,318** | — |
| LightGCN K=2 entraîné | 0,426 | 0,311 | — |

**Lectures clés** :
1. **Strict articles** : B3-e champion, LightGCN entraîné l'égale (gagne NDCG/MRR).
2. **Étendu articles** : PPR α=0,85 champion (mais sweep K_in a révélé un NOUVEAU champion non-encore intégré au tableau public : `PPR s=5 jp_only α=0,70` à M1=0,516, voir §3.5).
3. **JP** : LightGCN nouveau champion toutes méthodes confondues.
4. **M3 inversion** : B2-a cosine pur est champion M3 sur les deux modalités. PPR α=0,95 (champion M1 étendu) est le PIRE en M3 (0,214). « Optimiser M1 ne gagne pas la pertinence sémantique. »

### 3.3 Détail Chantier 4 — LightGCN

- **Design A** : propagation BGE-M3 sur graphe citations Art↔JP. Questions = users BGE-M3 figés. Scoring **cosinus** (sinon biais norme/hub effondre). K=2 (over-smoothing visible K0<K1<K2>K3).
- **Entraînement** : BPR cosinus τ=0,1 + ancrage `λ‖E0−BGE‖²` (λ=1,0). Sans ces 2 corrections, drift sur 840 paires et s'effondre.
- **Validation multi-seed (3 seeds : 1, 2, 42)** :
  - M1 strict = 0,704 ± 0,007 (0,698 / 0,712 / 0,701)
  - Δ vs untrained = +0,051 (toujours positif, ∈ [+0,045 ; +0,059])
  - NDCG strict = 0,532 ± 0,003
- **Caveat** : conflate apprentissage + changement d'init des non-embeddés (zéro → aléatoire). À durcir par ablation init.
- **Tradeoff entraînement** : à K=2, échange un peu de JP (0,437 → 0,426, -0,011) contre beaucoup de précision articles (+0,051). Les 840 paires sont des positifs articles uniquement, zéro supervision JP.
- **Pas de PyTorch Geometric** : propagation pure PyTorch sur matrice de citations sparse. CPU+MPS, ~5 min.
- **Référence script** : `05-Technique/benchmark/etape1_embedding_pur/scripts/31_lightgcn.py`
- **Handoff détaillé** : `01-Projet/handoffs/handoff-LightGCN-v2-2026-06-08.md`

### 3.4 Détail M3 — LLM-as-judge

- **Modèle** : Gemma 4 26B (`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` rév. 519bdca117c8) sur cluster Hector node L40S 46 Go.
- **Prompt** : v2 retenu (v1 trop strict, GT à 45% n2 ; v2 à 78% n2). Cache hashé sur (prompt+modèle) → édition propre.
- **Run complet** : 46 633 jugements sur 9 méthodes, ~40 min wall-clock (~16 jugements/s).
- **Validation 3 contrôles** :
  - SANITY GT-singleton : 78% n2 (1452/1864) — la GT est massivement créditée.
  - Test discriminant : random hors-GT → **0% n2** (la métrique discrimine).
  - Audit croisé Gemma vs Opus (36 items) : kappa pondéré 0,64 (accord substantiel), exact 61%, ±1 niveau 97%.
- **Dénominateur** : `k_fixed` (2K=20). Alternative `k_ranking` à arbitrer Johnny — n'impacte que B3-e/B4-d. Switch = re-agrégation seule, zéro re-jugement.
- **Référence script** : `05-Technique/benchmark/etape1_embedding_pur/scripts/23_eval_m3_llm_judge.py`
- **Handoff détaillé** : `01-Projet/handoffs/handoff-M3-COMPLET.md`

### 3.5 Détail sweep PPR (s × seed × α)

48 configurations testées (s ∈ {5,10,20,50} × seed ∈ {art_only, jp_only, both} × α ∈ {0,5; 0,7; 0,85; 0,95}). Norm row uniquement (sym collapse à cosine vérifié).

**Découvertes** :
- **Champion étendu nouveau** : `PPR s=5 jp_only α=0,70` → M1 ext = 0,516 (vs ancien 0,421, **+22%**), NDCG ext = 0,508.
- **Seed `jp_only` domine étendu** : les JP sont des hubs informatifs, les articles ne le sont pas (art_only inutilisable côté JP, M1_jp = 0,047).
- **K_in petit (5) > grand (50)** sur étendu.
- **α=0,50 gagne sur les métriques de rang** (MRR, NDCG strict).

**Fichier d'analyse complet** : `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/ppr_kin_sweep_analysis.md`

**Pas encore intégré au tableau global** : `24_build_global_table.py` n'a pas la nouvelle ligne. À faire si le user veut.

---

## 4. Présentation Week-10 (le deck Johnny)

### 4.1 Le fichier

- **Source** : `01-Projet/presentations/Week-10-2026-06-09-Presentation-Superviseur.tex` (25 pages)
- **PDF** : même nom .pdf à côté
- **Couleurs custom** : `accent` (bleu), `accentsoft` (bleu clair), `ok` (vert), `warn` (rouge), `artcol`, `jpcol`, `crosscol`
- **Macros** : `\keybox{}`, `\warnbox{}`, `\okbox{}`, `\crossbox{}` pour callouts colorés
- **Compilation** :
  ```bash
  cd 01-Projet/presentations
  /usr/local/texlive/2025/bin/universal-darwin/pdflatex \
    -interaction=nonstopmode -halt-on-error \
    Week-10-2026-06-09-Presentation-Superviseur.tex
  ```

### 4.2 Structure (25 pages)

| # | Section | Slides |
|---|---|---|
| 1-2 | Titre + plan | titlepage, ToC |
| 3-4 | Récap Week-9 baselines + constat sparsité GT (|GT|≈1) | 2 slides |
| 5-6 | 4 décisions Week-9 + roadmap actualisée 21 paliers | 2 slides |
| 7 | Plan d'attaque 4 chantiers (TikZ flowchart avec dépendances) | 1 slide |
| 8-9 | Chantier 1 — pourquoi métriques + détail (1/2) M1/M2/Hit | 2 slides |
| 10-11 | Chantier 1 — détail (2/2) MRR/NDCG/M3 + conventions+impl | 2 slides |
| 12-13 | Tableau global Articles + JP (avec LightGCN intégré) | 2 slides |
| 14 | PPR sweep α (sur s=10 fixé) | 1 slide |
| 15-17 | **Section M3** : tableau 9 méthodes, validité (3 contrôles), interprétation | 3 slides |
| 18-19 | Chantier 3 — enrichissement bench (pistes + pivot conceptuel) | 2 slides |
| 20-23 | **Section LightGCN G0** : one-pager, ablation, leçons techniques, limites+suite | 4 slides |
| 24 | Questions à arbitrer Johnny (7 questions, dont M3 lecture, dénominateur, bench enrichi) | 1 slide |
| 25 | Merci | 1 slide |

### 4.3 Deck LightGCN standalone (alternatif)

Un deck standalone existe aussi : `01-Projet/presentations/LightGCN-Resultats-G0-2026-06-08.tex` (7 frames focused). Compile aussi avec pdflatex.

---

## 5. Fichiers clés à connaître (par ordre d'importance pour Codex)

### 5.1 Code (scripts Python)

| Path | Rôle |
|---|---|
| `05-Technique/benchmark/etape1_embedding_pur/scripts/metrics.py` | Module panel métriques (5 fonctions + helpers) |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/18_eval_m1_m2.py` | Éval baselines B* (B2-a, B3-a, B3-e, B4-a/c/d/e/f), dump rankings.parquet |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/20_ppr_naive.py` | PPR row + sym, α sweep (s=10 fixé) |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/25_ppr_kin_sweep.py` | PPR sweep complet s × seed × α (48 configs) |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/31_lightgcn.py` | **LightGCN design A** : modèle, propagation, BPR cosinus+ancrage, eval, 5 variantes |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/23_eval_m3_llm_judge.py` | M3 LLM-judge (mono-item concurrent, cache write-through, --mock/--pilot/--methods/--denom) |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/24_build_global_table.py` | Construit le grand tableau global (CSV + Markdown) |
| `05-Technique/benchmark/etape1_embedding_pur/etape1/config.py` | Config paths centralisé (EMB_ARTICLES_ALL, JP_INDEX, GRAPH_NPZ, etc.) |

### 5.2 Données (gitignored, dans checkout principal uniquement)

| Path | Contenu | Taille |
|---|---|---|
| `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/bench_global.json` | 971 questions + GT | 3,5 MB |
| `…/data/global_bench/eval_m1_m2.csv` | éval B* panel complet | 2 MB |
| `…/data/global_bench/ppr_naive_eval.csv` | éval PPR α sweep | — |
| `…/data/global_bench/ppr_kin_sweep_eval.csv` | éval sweep complet s × seed × α | 7,1 MB |
| `…/data/global_bench/rankings.parquet` | rankings des 8 méthodes champions (entrée M3) | — |
| `…/data/global_bench/eval_m3.csv` | éval M3 par méthode | 557 KB |
| `…/data/global_bench/eval_m3_summary.json` | M3 agrégé | — |
| `…/data/global_bench/m3_judge_cache_2a34d2c5.csv` | cache jugements bruts M3 (réutilisable) | 13 MB |
| `…/data/global_bench/discriminant_test.json` | test discriminant M3 (random=0% n2) | — |
| `…/data/global_bench/global_table_{articles,jp}.csv` | grand tableau consolidé | — |
| `…/data/global_bench/global_table.md` | rendu Markdown | — |
| `…/data/global_bench/lightgcn_summary.json` | éval LightGCN (5 variantes K) | — |
| `…/data/global_bench/lightgcn_eval.csv` | éval LightGCN détaillée | — |

### 5.3 Documentation (committées dans git)

| Path | Rôle |
|---|---|
| `CLAUDE.md` / `AGENTS.md` | Instructions racine (identiques) |
| `01-Projet/handoffs/handoff-CODEX-2026-06-09.md` | **CE FICHIER** |
| `01-Projet/handoffs/handoff-SESSION-PRINCIPALE-2026-06-09.md` | Vue master des 4 chantiers + M3 |
| `01-Projet/handoffs/handoff-LightGCN-v2-2026-06-08.md` | Détails LightGCN (sessions, design, reproduction) |
| `01-Projet/handoffs/handoff-M3-COMPLET.md` | Détails M3 (run, validation, runbook cluster, pièges) |
| `01-Projet/handoffs/handoff-LightGCN.md` (v1) | Plan initial LightGCN, remplacé par v2 |
| `01-Projet/handoffs/handoff-M3-LLM-judge.md` | Plan initial M3, remplacé par COMPLET |
| `01-Projet/handoffs/handoff-M3-collecte-agregation.md` | Plan récolte M3, remplacé par COMPLET |
| `01-Projet/journal/2026-06-08.md` | Journal détaillé du jour (panel + tableau + M3) |
| `01-Projet/journal/2026-06-03.md` | Debrief point Johnny Week-9 |
| `01-Projet/journal/2026-05-28.md` | Sessions précédentes (PPR, B4 variants) |
| `01-Projet/decisions/ADR-001-Versionnage-Graphe-G0-Vn.md` | Stratégie versionnage graphe + régimes supervision |
| `02-Etat-de-l-art/gnn/LightGCN-2020.md` | Fiche papier LightGCN |
| `02-Etat-de-l-art/recommandation/Johnny-LightGCN-Solution-Notebook.md` | Fiche notebook envoyé par Johnny |
| `06-Analyses/syntheses/LightGCN-Resultats-G0-2026-06-08.md` | Synthèse résultats LightGCN (slide-ready) |
| `06-Analyses/syntheses/sota-gnn-reco-2026.md` | Mémo SOTA GNN reco |
| `01-Projet/presentations/Week-10-…tex/.pdf` | **Deck du point Johnny aujourd'hui** |
| `01-Projet/presentations/LightGCN-Resultats-G0-…tex` | Deck standalone LightGCN |

---

## 6. Comment lancer / reproduire les choses

### 6.1 Environnement Python

```bash
# Activer venv dédié (CRÉÉ DANS LE CHECKOUT PRINCIPAL UNIQUEMENT)
PY="/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/.venv/bin/python"
$PY --version  # Python 3.12.x avec torch 2.12 + numpy, scipy, pandas, pyarrow
```

### 6.2 Re-générer les résultats des baselines

```bash
S="/Users/.../legal_knowledge_graph/05-Technique/benchmark/etape1_embedding_pur/scripts"
$PY $S/18_eval_m1_m2.py            # B* (panel complet, ~100s)
$PY $S/20_ppr_naive.py             # PPR α sweep (s=10, ~10 min)
$PY $S/24_build_global_table.py    # consolide les CSV
```

### 6.3 Lancer LightGCN

```bash
$PY -u $S/31_lightgcn.py           # entraînement K=2 (~5 min CPU+MPS)
NOTRAIN=1 $PY $S/31_lightgcn.py    # diagnostic seul (untrained), ~70s
SEED=1 $PY $S/31_lightgcn.py       # autre seed (validation multi-seed)
TRAIN_K=1 $PY $S/31_lightgcn.py    # entraîner à un autre K
```

Note : les runs écrivent `lightgcn_eval.csv` (clobber). Pour multi-seed propre, sauvegarder/suffixer.

### 6.4 Compiler la présentation

```bash
PDFLATEX="/usr/local/texlive/2025/bin/universal-darwin/pdflatex"
cd /Users/.../legal_knowledge_graph/01-Projet/presentations
$PDFLATEX -interaction=nonstopmode -halt-on-error \
  Week-10-2026-06-09-Presentation-Superviseur.tex
```

### 6.5 M3 LLM-judge (cluster Hector — pour info)

Procédure complète dans `handoff-M3-COMPLET.md` §5. Résumé :
- SSH `kaeppelin-22@<NODE>` (DEMANDER la node, varie ; node57 le 2026-06-08)
- `LKG_REPO=$HOME/legal_knowledge_graph`
- Env prêt (vllm 0.19.0, openai 2.32.0) — **PAS de pip install**
- Pilote : `./run_m3_judge_on_cluster.sh pilot` puis inspecter SANITY
- Full : `./run_m3_judge_on_cluster.sh full` (~45 min Gemma 4 26B sur L40S)

---

## 7. Pièges et conventions techniques (appris dans les sessions précédentes)

### 7.1 Pièges scientifiques

1. **Sanity check obligatoire** : toute nouvelle méthode reproduit une baseline connue avant interprétation. Ex : LightGCN cosine_raw doit valoir 0,459 (= B2-a) à K=0.
2. **Scoring cosinus pour méthodes propagées** : sinon biais de norme (les hubs gonflent), strict + JP s'effondrent (M1 ≈ 0,02). L2-normaliser après propagation.
3. **Anti-leak** : la cohorte 971 ne doit JAMAIS être vue à l'entraînement (assert dans le code).
4. **GT vide → NaN**, pas 0 : pour ne pas biaiser la moyenne sur les questions singleton non-couvertes.
5. **MRR cappé à K**, pas full ranking : cohérent avec le reste du panel, évite de calculer le ranking complet.
6. **NDCG binaire pour l'instant** (rel ∈ {0,1}). Multi-niveau viendra avec enrichissement GT (Chantier 3).

### 7.2 Pièges techniques

1. **Notation `s` vs `K_in`** : on a propagé `K_in → s` (seed size) partout pour ne pas confondre avec le K=10 du Recall@K. **Toute nouvelle doc doit utiliser `s`**.
2. **PyG inutile** : LightGCN tourne en PyTorch pur sur matrice sparse. Pas la peine d'installer torch_geometric.
3. **Pas de pip install non-pinné** sur cluster Hector (env --user fragile).
4. **stdout python bufferisé sous nohup** : suivre la progression via le **cache write-through** (`wc -l`), pas le stdout.
5. **vLLM met ~8 min à charger** (14 GB AWQ). Pour itérer un prompt, garder vLLM persistant.
6. **openrsync (macOS) n'a pas l'anchor `/./`** : utiliser `(cd <base> && rsync -aR chemin/relatif <dest>)`.
7. **SSH cluster gaté Slurm** (pam_slurm_adopt) : on ne peut SSH sur la node QUE pendant une allocation active. Rapatrier AVANT la fin du job.
8. **pdflatex non dans PATH** : utiliser chemin absolu `/usr/local/texlive/2025/bin/universal-darwin/pdflatex`.

### 7.3 Pièges Git / structure

1. **Données + .venv gitignored** dans le checkout principal, absents des worktrees.
2. **Scripts hardcodent REPO** en chemin absolu → fonctionnent depuis n'importe où, lisent/écrivent le checkout principal.
3. **Tout est sur `etape1-embedding-pur`** depuis le 9 juin. Branche `main` reste antérieure.
4. **Plusieurs handoffs M3** existent : `handoff-M3-LLM-judge.md` (plan), `handoff-M3-collecte-agregation.md` (récolte), **`handoff-M3-COMPLET.md` (master, à lire)**. Idem LightGCN : v1 → v2 (à lire).

---

## 8. Questions ouvertes / décisions Johnny en attente

Pour le point Johnny **9 juin midi** (peut-être déjà passé selon quand tu lis ce handoff), les 7 questions à arbitrer (slide 24 de Week-10) :

1. **Validation panel métriques** : Hit@K + MRR@K cappé + NDCG@K binaire OK ?
2. **Lecture M3** : acter que cosine pur est champion M3 ? Implication forte sur stratégie GNN.
3. **Dénominateur M3** : `k_fixed` (actuel, 2K=20) vs `k_ranking` (taille réelle) ? N'impacte que B3-e/B4-d.
4. **Périmètre tableau global** : inclure LLM seul / LLM+RAG dès maintenant ?
5. **Enrichissement bench** : feu vert reverse-engineering décisions (plusieurs semaines) ou rester sur visa + Légifrance ?
6. **Fichier Mattermost** : LightGCN est fait, donc cette question est résolue.
7. **Cible Étape 1 vs Étape 2** : frontière à acter.

---

## 9. Prochaines étapes (par priorité)

### 9.1 Si Johnny valide tel quel (scénario nominal)

1. **Intégrer le nouveau champion PPR** au tableau global : `PPR s=5 jp_only α=0,70` côté articles étendu. Modifier `24_build_global_table.py` pour ajouter cette ligne en plus de PPR α=0,85/0,95. ~30 min.
2. **Ablation init LightGCN** : isoler le `+0,051` entraîné en testant `init zéro` (comme untrained) vs `init aléatoire` à K=2. ~1h.
3. **Régime D supervision JP** : injecter des paires positives JP+article au train pour ne plus éroder le JP. Cf. ADR-001. ~1 jour.
4. **G0 → V1** : retirer les 41 824 nœuds morts du pool, re-mesurer toutes les méthodes. Quantifie l'apport du nettoyage. ~1 jour.
5. **Chantier 3 démarrage** : commencer par expansion GT via visa des arrêts (gratuit, ancré juridiquement). ~2-3 jours.

### 9.2 Si Johnny veut élargir

- **LLM seul (Gemma) + LLM+RAG** : parser articles cités dans la génération libre (regex `art\. \d+ du Code [a-z ]+`). Handoff dédié à créer. ~2 sessions.
- **Variantes GNN** : R-GCN inductif, HGT, GraphSAGE après LightGCN. ~3-4 semaines cumulés.

### 9.3 Backlog non-priorisé

- M3 sur LightGCN (rankings dumpés OK, juste lancer le run cluster).
- M3 sur LLM seul / LLM+RAG (handling parsing spécial).
- Tuning hyperparams LightGCN (τ, λ, lr, epochs).
- Sweep K_in PPR plus fin (s=2, s=3, α=0,4, etc.).
- Q-PPR (ajouter question comme nœud, restart depuis Q).

---

## 10. Comment travailler en tant que Codex sur ce projet

### 10.1 Conventions de communication

- **En français** avec accents (toujours).
- **Pas de copier-coller terminal** : fournir des `.sh` ou exécuter via Bash. Ne JAMAIS demander à l'utilisateur de lancer une commande copiée-collée.
- **Vérifier avant de conclure** : test discriminant, sanity check, re-run baseline connue. La mémoire de l'utilisateur sur les runs passés est fiable, ne pas la contredire sans preuve.
- **Réponse courte par défaut**, expansion si demandé. Le user préfère narrer brièvement ce qu'on fait avant d'agir.

### 10.2 Workflow typique

1. **Lire** le handoff master (ce fichier) + le journal du jour.
2. **Annoncer** ce qu'on va faire en 1 phrase.
3. **Agir** (Bash, Edit, Write).
4. **Vérifier** (compile, run, lire sortie).
5. **Restater** en clair ce qui a changé + ce qui reste.

### 10.3 Quand le user dit « commit »

- Stager les fichiers ciblés (pas `git add .`).
- Exclure `.DS_Store`, `.aux/.nav/.out/.snm/.toc/.log` (LaTeX), `__pycache__/`.
- Inclure le PDF compilé (livrable).
- Message commit avec Co-Authored-By trailer.
- Push sur `etape1-embedding-pur`.

### 10.4 Si on est bloqué

- Re-lire le handoff (souvent la réponse est dedans).
- Lire les sous-handoffs (LightGCN v2, M3 COMPLET).
- Lire le journal du jour.
- Si vraiment bloqué : demander au user de pointer le bon fichier plutôt qu'explorer à l'aveugle.

---

## 11. Mémoires utilisateur (résumé)

Mémoires persistantes Claude (source : `~/.claude/projects/-Users-matthieu-kaeppelin-Documents-5-Pro-Stages-FE-recherche-legal-knowledge-graph/memory/MEMORY.md`). Si Codex a son propre système de mémoire, ces faits méritent d'y être enregistrés.

- **Pas de copier-coller depuis le terminal** : toujours .sh / Bash direct.
- **Env --user cluster fragile** : pip install non-pinné casse l'env partagé.
- **Vérifier sur preuve** : test discriminant avant de conclure.
- **doctrine_qgen** : 4/19 sections dépassent contexte 32k Gemma 4 (à investiguer).
- **legi.py sans hunspell** : `pip install --no-deps legi` + libarchive-c + appdirs.
- **Décisions Johnny Week-9** : panel métriques complet, grand tableau LLM/RAG/cosine/PPR/GNN, LightGCN.
- **Données réelles LightGCN** : graphe = graph_penal.npz [118k JP × 87k art], emb articles 31357/87821, JP 116755.
- **M3 LLM-judge implémenté** : run complet fait, 9 méthodes.
- **Accès cluster Hector** : SSH kaeppelin-22@<NODE> (DEMANDER la node), LKG_REPO, env prêt sans pip.
- **Décisions Johnny Week-9** : panel métriques (M1/M2/M3/Hit@K + MRR/NDCG), grand tableau global, LightGCN à tester.

---

## 12. Référence courte — où aller pour quoi

| Je veux... | Aller voir |
|---|---|
| Comprendre les chantiers | `01-Projet/handoffs/handoff-SESSION-PRINCIPALE-2026-06-09.md` |
| Voir le deck Johnny | `01-Projet/presentations/Week-10-2026-06-09-Presentation-Superviseur.pdf` |
| Comprendre LightGCN | `handoff-LightGCN-v2-2026-06-08.md` + `scripts/31_lightgcn.py` + `06-Analyses/syntheses/LightGCN-Resultats-G0-2026-06-08.md` |
| Comprendre M3 | `handoff-M3-COMPLET.md` + `scripts/23_eval_m3_llm_judge.py` |
| Voir les résultats du jour | `01-Projet/journal/2026-06-08.md` |
| Voir le grand tableau | `data/global_bench/global_table.md` |
| Comprendre le sweep PPR | `data/global_bench/ppr_kin_sweep_analysis.md` |
| Comprendre ADR versionnage graphe | `01-Projet/decisions/ADR-001-Versionnage-Graphe-G0-Vn.md` |
| Connaître le panel métriques | `scripts/metrics.py` |
| Lancer / re-générer | `§6` de ce handoff |

---

## Fin

Ce handoff vise à donner à Codex (ou tout agent reprenant le projet) **tout le contexte pour avancer sans question** sur les ~3 dernières semaines de travail intensif (Week-9 → Week-10). Si certaines parties manquent de clarté, ne pas hésiter à demander à l'utilisateur de pointer le bon fichier — il est très investi et connaît bien sa structure.

**Le projet est mûr scientifiquement** : 5 méthodes testées sur 6 métriques, grand tableau consolidé, 2 chantiers majeurs (LightGCN + M3) livrés avec validation rigoureuse. **Le rythme attendu** est une session productive (2-4h) par jour avec un point superviseur hebdomadaire.

Bonne reprise !
