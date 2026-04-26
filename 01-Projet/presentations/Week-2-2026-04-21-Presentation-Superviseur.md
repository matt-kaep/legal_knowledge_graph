---
tags: [presentation, projet, superviseur, benchmarks, graphes, semaine-2]
type: presentation
semaine: 2
date: 2026-04-21
audience: superviseur-de-stage
format: slides-markdown
status: en-cours
modified: 2026-04-21
---

# Présentation Superviseur — Semaine 2 (14-21 avril 2026)

> [!info] Objectif de la présentation
> Montrer où l'on en est **dans le plan global benchmark M1-M6 + configurations B1-B4 + C** arrêté en Week 1, en rendant compte de : l'**exploration**, les **tests** menés, les **partis pris**, et les **benchmarks effectivement constitués cette semaine**.

---

## Plan

1. Rappel du cadre consolidé fin Week 1
2. Shift stratégique du 16 avril : extension plutôt que from-scratch
3. Exploration des sources (14-19 avril)
4. Tests et validations (biais circulaire, parse rates, orchestration)
5. Partis pris méthodologiques
6. **Livrable 1** — Benchmark CRFPA (niveau M1-M2)
7. **Livrable 2** — Benchmark Rapprochements (niveau M3-M4)
8. **Livrable 3** — Graphes Phase A & B (infrastructure pour C)
9. Découverte méthodologique émergente (renumérotation 2016)
10. Articulation avec la littérature et M6 existant
11. Bilan + prochaines étapes (jalon 16 mai)
12. Points de décision pour le superviseur

---

## Slide 1 — Rappel : architecture cible du benchmark (Week 1)

Design consolidé le 16 avril dans [[Benchmark-KG-Juridique-FR-Design]] :

**6 modules couvrant les deux niveaux de raisonnement juridique**

| Niveau | Modules | Tâche |
|---|---|---|
| **Niveau 1** — Principe | M1 (Principe QA), M3 (Multi-saut), M4 (QCM), M5 (Temporalité) | "Quelles sont les conditions de X ?" |
| **Niveau 2** — Application | M2 (Applied QA), M6 (Interprétation arrêt ⭐) | "Dans mon dossier, X est-il valable ?" |

**5 configurations à benchmarker sur chaque module**

| # | Configuration |
|---|---|
| B1 | LLM seul (zero-shot) |
| B2 | LLM + RAG vectoriel |
| B3 | LLM multi-step / agentique (sans RAG) |
| B4 | LLM + RAG + agentique |
| **C** | LLM + **GraphRAG** — notre cible |

> Référence : [[Benchmark-KG-Juridique-FR-Design]] — Week 1 consolidé.

---

## Slide 2 — Shift stratégique du 16 avril

Après lecture exhaustive des benchmarks FR existants ([[Alhajar-2025-Les-Audits-Affaires]], [[Harvard-LIL-2024-Open-French-Law-RAG]], [[Louis-2022-BSARD]], [[Guha-2023-LegalBench]], [[Zheng-2021-CaseHOLD]]) :

### Décision

> **Ne pas construire from-scratch un benchmark FR — étendre Les-Audits-Affaires** (LegMLAI, juin 2025) qui fournit déjà 2 670 cas sur 9 codes juridiques.

### Ce qui change

- **M1 = Les-Audits-Affaires tel quel** (2 670 cas déjà disponibles)
- **Gain** : ~6 mois sur la construction d'un benchmark articles de loi

### Ce qui reste à construire sur mesure

- **M2, M3, M4, M5, M6** — pas de benchmark français existant pour ces niveaux
- Surtout **M6 (interprétation contextualisée)** : inédit dans la littérature FR

---

## Slide 3 — État début de semaine 2 (14 avril)

**Acquis** (Week 1) :
- Design benchmark 6 modules + 5 configs arrêté
- Formats canoniques définis : [[Format-Fondement-Juridique]] (`code_civil:1240`), [[Format-Jurisprudence]] (ECLI, `sens_vs_question`)
- Base Judilibre enrichie disponible (553 k arrêts CC + 430 k CA + 142 k TJ avec `code_article_pairs` déjà extraits)
- 5 cas gold M6 MVP construits (cf. `m6_mvp/gold_cases.py`)

**Manque critique** :
- **Pas de vraie ground truth produite** pour les modules M1-étendu, M3, M4
- Aucun graphe construit — M3 et M5 bloqués

> Objectif semaine 2 : **produire de la GT réelle** + **poser l'infrastructure du graphe**.

---

## Slide 4 — Exploration (14-19 avril)

### Sources CRFPA découvertes

> Depuis **2025**, le **Conseil National des Barreaux** publie les **grilles de notation officielles** de ses épreuves écrites. Inédit — suite à un recours administratif.

- **Sujets CNB** 2024 + 2025 (12 épreuves × 2 ans = 24 énoncés canoniques)
- **Grilles officielles** = définissent les points attendus par le jury
- **Meilleures copies IEJ Strasbourg** (sessions 2022-2025)

### Sources Judilibre exploitables

Dans les records `Cour de cassation` de la base enrichie :
- Champ `rapprochements` : **autres arrêts à mettre en regard** selon la Cour elle-même
- Champ `code_article_pairs` : articles cités, déjà au format canonique `code_civil:1240`
- Sur 553 075 arrêts CC : 63 695 ont des rapprochements (11,5 %)

### Doctrine (reportée Phase 2)

- Actu-Juridique, Village-Justice, Le Petit Juriste : explorables mais variables en qualité
- Dalloz-Actualité, HAL-SHS : bloqués par protections anti-bot
- → pas prioritaires pour le MVP

> Inventaire complet : [[Sources-Benchmark-MVP]]

---

## Slide 5 — Tests et validations menés

### Test 1 — Orchestration multi-LLM

Pipeline d'extraction CRFPA : `Sujet + Grille + Copie → Opus → Rubrique JSON`.
- **10 subagents Opus en parallèle** (1 par matière)
- Taux de réussite au premier coup : **50 %** (oubli du Write final)
- Rattrapage via SendMessage explicite → 100 %
- **Temps total** : 6 h d'orchestration pour 11 matières

### Test 2 — Détection d'un biais circulaire

**V1 "droit des obligations 2024"** : 100 % des rubriques étaient **LLM-intrinsèques** (le modèle inventait les réponses à partir du sujet seul). **Biais circulaire détecté** avant dissémination.

Correction : v2 avec triangulation **sujet + grille CNB + copies IEJ** → **8 % de points LLM-intrinsèques** seulement.

### Test 3 — Parse rate des rapprochements

- Sur échantillon récent (2020-2024) : **80 %** des rapprochements sont parsables par regex (numéro de pourvoi extractible).
- Sur corpus complet : **16,3 %** seulement. Les titres pré-1980 n'ont pas de format standardisé ("Bulletin 1956, IV, n° 604"). Limite de la donnée brute, pas de la regex.

### Test 4 — Visualisation

Spring-layout sur 31 k nœuds → 10 min CPU. Acceptable en préparatoire, prohibitif en temps réel.

---

## Slide 6 — Partis pris méthodologiques

### Parti pris 1 — Format de la ground truth : **rubrique hiérarchisée**

Inspiré de HealthBench (OpenAI 2025) et PLawBench (Shi et al. 2026).

| Strate | Poids | Signification |
|---|---|---|
| `core` | 3 | Un étudiant L2 doit les citer |
| `expected` | 2 | Attendu d'un praticien |
| `expert` | 1 | Navigation KG profonde |

Score composite = `(3·cov_core + 2·cov_expected + cov_expert) / total_weight`.

Discrimine structurellement : RAG vectoriel plafonne sur `core + expected`, GraphRAG doit décrocher sur `expert`.

### Parti pris 2 — Triangulation anti-biais

Sujet CNB + Grille CNB + Copies IEJ = référence sans dépendance LLM sur la connaissance substantielle. Évite le cercle "LLM génère la GT et évalue un LLM".

### Parti pris 3 — Exploitation des rapprochements institutionnels

Champ `rapprochements` Judilibre → ground truth **gratuite et officielle** pour la tâche "navigation entre JP liées". Aucun benchmark FR ne l'exploite.

### Parti pris 4 — MVP vite, scale ensuite

Privilégier **20-50 questions haute qualité + vérification manuelle** avant d'engager un scale-up à 300+. Démarrer les baselines tôt.

---

## Slide 7 — Livrable 1 — Benchmark CRFPA (niveau M1-M2)

### Positionnement dans l'architecture

Couvre les niveaux **Principe QA** et **Applied QA** sur un corpus **examinal** (sujets d'examen d'avocat).

### Chiffres

| Métrique | Valeur |
|---|---:|
| Questions | **38** sur 11 matières CRFPA 2025 |
| Points de rubrique | 681 (294 core + 181 expected + 206 expert) |
| Articles uniques | 296 |
| JP uniques | 188 |
| Sources | 52 % grille_cnb · 40 % copie_iej · 8 % llm_intrinsic |

### Matières couvertes

Obligations · Civil · Pénal · Affaires · Social · Administratif · Fiscal · International/Européen · Procédure civile · Procédure pénale · Procédure administrative.

### Limites identifiées

- 100 % des JP portent une `uncertainty_note` → validation Judilibre requise avant usage en scoring.
- Copies IEJ = scans manuscrits → `tesseract-fra` requis pour passer à l'échelle.

> Détails : [[Design-Rubrique-Hierarchisee]], [[Recap-MVP-2026-04-20]].

---

## Slide 8 — Livrable 2 — Benchmark Rapprochements (niveau M3-M4)

### Positionnement dans l'architecture

Couvre **M3 (multi-saut)** et **M4 V4 (détection de revirement)** dans leur version *retrieval de JP liée*. **Aucun benchmark FR existant** ne teste cette tâche.

### Tâche évaluée

Étant donné un arrêt Cass (référence + texte intégral) → lister les **rapprochements officiels** que la Cour elle-même a déclarés.

### Pipeline

Script monolithique 3 passes streaming sur le fichier JSONL de 5 Go :
1. Scan + filtrage (≥3 rapprochements parsables par pourvoi).
2. Index inverse `pourvoi_normalisé → id Judilibre`.
3. Résolution + stratification par chambre + export.

Durée totale : **< 1 minute**.

### Chiffres

| Métrique | Valeur |
|---|---:|
| Arrêts CC scannés | 553 075 |
| Questions produites | **1 532** |
| Rapprochements en GT | 5 832 |
| Taux de résolution vers ID | **98,7 %** |

### Stratification par chambre

Soc. 375 · Crim. 370 · Civ. 2e 291 · Civ. 1re 236 · Civ. 3e 181 · Com. 39 · Ass. plén. 26 · Ch. mixte 10.

> Détails : [[Design-Benchmark-Rapprochements]].

---

## Slide 9 — Les deux benchmarks sont complémentaires

| Axe | Benchmark CRFPA (M1'-M2) | Benchmark Rapprochements (M3-M4) |
|---|---|---|
| Tâche | QA juridique structurée | Recommandation de JP liées |
| Ground truth | Rubrique CNB + copies IEJ | Rapprochements Cass officiels |
| Mesure | Qualité du raisonnement | Navigation entre décisions |
| **Discriminant cible** | RAG vs LLM seul | **Graphe vs RAG vectoriel** |
| Taille | 38 Q (qualité max) | 1 532 Q (volume + officiel) |
| Spécialité | Ancré *pédagogie juridique* | Ancré *pratique judiciaire* |

> Ensemble, ces deux benchmarks couvrent la **matière examinale** et la **matière judiciaire effective** — complément naturel d'un éventuel M1 via Les-Audits-Affaires (business law).

---

## Slide 10 — Livrable 3a — Graphes Phase A (JP-JP seuls)

### Motivation

Puisqu'on a extrait 18 901 rapprochements parsables, que donne leur structure graphe ? C'est la première brique de l'infrastructure pour M3, M5 et la config C.

### Résultats par périmètre

| Périmètre | Nœuds | Arêtes | Composantes | Plus grosse CC |
|---|---:|---:|---:|---:|
| resserre (benchmark + cibles) | 5 618 | 5 436 | 884 | 120 |
| large (tous arrêts rapp) | 21 534 | 18 201 | **5 199** | 1 725 (8 %) |
| tout_cc | 534 600 | 18 201 | 518 265 | 1 725 |

### Lecture

- **Tissu jurisprudentiel très fragmenté** : 5 199 composantes disjointes.
- Degré max = 13 (pas de super-hubs — contraste avec les citations scientifiques).
- La Cour tisse des **lignées locales thématiques**, pas un réseau transversal.

### Top 3 arrêts-pivots (PageRank)

1. Civ. 3e, 23 sept. 2009, n° 07-20.965
2. Civ. 3e, 17 déc. 2014, n° 13-19.582
3. Soc., 25 nov. 2015, n° 14-18.821

---

## Slide 11 — Livrable 3b — Graphes Phase B (bipartite JP × Articles)

### Apport structurel

Ajout des nœuds Article (format `pair_key` issu de `enrichissement_base_complete.ipynb`) et des arêtes `cite` (Decision → Article) depuis `code_article_pairs`.

### Résultats

| Périmètre | JP | Articles | Arêtes cite | Arêtes rapproche | Composantes | Plus grosse CC |
|---|---:|---:|---:|---:|---:|---:|
| resserre | 5 618 | 3 975 | 18 366 | 5 958 | **16** | **9 586 (99,9 %)** |
| large | 21 534 | 9 940 | 73 168 | 18 201 | **31** | **31 390 (99,7 %)** |
| tout_cc | 534 600 | 33 343 | 1 196 681 | 18 201 | quasi-connexe | 493 721 (92 %) |

### Observation centrale

> **5 199 composantes → 31** en ajoutant les articles.
> Plus grosse CC : 8 % → **99,7 %** des nœuds.

**Les articles sont les ponts qui unifient le tissu jurisprudentiel.** Deux arrêts sans rapprochement explicite deviennent connectés en 2 sauts par un article commun. Démonstration empirique du besoin d'un graphe bipartite pour le KG final.

### Top 5 articles hubs

| pair_key | Citations | Domaine |
|---|---:|---|
| `code_de_procedure_civile:700` | 12 268 | Frais irrépétibles |
| `code_de_procedure_civile:455` | 2 837 | Motivation des jugements |
| `code_de_l_organisation_judiciaire:R431-5` | 2 553 | Composition |
| `code_civil:1134` | 1 992 | Force obligatoire (avant 2016) |
| `code_civil:1382` | 1 030 | Responsabilité (avant 2016) |

> Détails : [[Design-Graphes-Phase-AB]].

---

## Slide 12 — Démonstration visuelle

Contraste **A1 ↔ B1** (même périmètre, 1 532 arrêts-sources) :

- **A1 — rapprochements seuls** : galaxie fragmentée, 884 composantes éparpillées, pas de centre.
- **B1 — bipartite JP × Articles** : noyau rouge dense au centre (articles hubs), JP en périphérie, structure centre-périphérie claire.

> À insérer côte-à-côte dans la slide :
> `data/figures/A1-rapp-resserre.png` | `data/figures/B1-bip-resserre.png`

6 figures produites au total (200 DPI, utilisables pour le mémoire).

---

## Slide 13 — Découverte méthodologique émergente

Le graphe a révélé un **problème d'identité temporelle des articles** :

| Ancien (avant 2016) | Nouveau (depuis 2016) | Règle |
|---|---|---|
| `code_civil:1382` | `code_civil:1240` | Responsabilité délictuelle |
| `code_civil:1134` | `code_civil:1103` / `1104` / `1193` | Force obligatoire |
| `code_civil:1147` | `code_civil:1231-1` | Responsabilité contractuelle |

### Conséquence

Deux arrêts sur la **même règle juridique** publiés avant/après 2016 ne sont pas connectés dans le graphe — faux clivage.

### Implication pour le M5 (Temporalité)

La résolution de ce problème est un **prérequis** pour M5, qui teste justement *"quelle était la règle à la date X"*. Décision d'implémentation recommandée : ajouter des arêtes `recodified_as` (option b) — fidélité historique préservée + navigabilité augmentée.

> Noté dans [[Format-Fondement-Juridique]] §7.

---

## Slide 14 — Articulation avec la littérature et M6 existant

### Couverture par livrable

| Module | Statut fin Week 2 | Source |
|---|---|---|
| **M1** — Principe QA | ✅ *Les-Audits-Affaires* (2 670 cas FR) | [[Alhajar-2025-Les-Audits-Affaires]] |
| **M1'** — CRFPA (étendu) | ✅ *livré* (38 Q, triangulé) | CNB + Cap'Barreau + IEJ |
| **M2** — Applied QA | 🔨 cases Hector synthétiques + construction manuelle |
| **M3** — Multi-saut | ✅ *Benchmark Rapprochements* (1 532 Q) | Judilibre rapprochements |
| **M4** — QCM (V4 revirement) | 🔨 variante auto depuis graphes Phase B | Graphes Phase B |
| **M5** — Temporalité | 🔨 nécessite résolution renumérotation 2016 | Noté §7 Format-Fondement |
| **M6** — Interprétation arrêt | ✅ *5 cas gold MVP* (16 avril) | [[Hector-AI-Benchmark-Interne]] |

### Contributions originales vs littérature

- **CRFPA** : première extension d'un benchmark FR sur corpus *examinal CNB officiel* avec triangulation.
- **Rapprochements** : aucun autre benchmark FR n'exploite le champ `rapprochements` Judilibre. Gap comblé.
- **Graphe bipartite** : apport empirique de la structure articles-ponts ; démonstration visuelle directe utilisable en mémoire.

---

## Slide 15 — Bilan semaine 2

### Livrables consolidés

```
05-Technique/benchmark/
├── data/rubrics/                     ← 38 Q CRFPA (M1'), 11 matières
├── data/rapprochements/              ← 1 532 Q rapprochements (M3)
├── data/graphs/                      ← Phase A : 3 périmètres (pkl + graphml)
├── data/graphs_bipartite/            ← Phase B : 3 périmètres (infra pour C)
├── data/figures/                     ← 6 PNG 200 DPI
├── build_rapprochement_benchmark.py
├── build_rapprochement_graphs.py
├── build_bipartite_graphs.py
└── visualize_graphs.py
```

### Chiffres consolidés

- **553 k** arrêts CC scannés
- **18 900** arêtes `rapproche` parsables
- **1,2 M** arêtes `cite` (Decision → Article)
- **33 k** articles canoniques uniques
- **38 + 1 532 = 1 570 questions** de benchmark à la fin de la semaine

### Docs Obsidian produits

- 2 journaux ([[2026-04-20]], [[2026-04-21]])
- 3 design docs ([[Design-Rubrique-Hierarchisee]], [[Design-Benchmark-Rapprochements]], [[Design-Graphes-Phase-AB]])
- 1 récap ([[Recap-MVP-2026-04-20]]) + 1 inventaire ([[Sources-Benchmark-MVP]])
- [[Format-Fondement-Juridique]] + §7 empirique

---

## Slide 16 — Prochaines étapes

### Semaine 3 (22-28 avril)

- **Phase B1 baselines** : LLM seul sur CRFPA + Rapprochements (Gemma 3, Mistral, Saul-Instruct, GPT-5, Claude Opus).
- **Phase B2 baselines** : LLM + RAG vectoriel Légifrance avec `multilingual-e5-large`.
- **Script de validation Judilibre** sur les 188 JP du benchmark CRFPA.
- **Générateur M4 V4** (QCM revirement) depuis graphes Phase B.

### Semaines 4-5

- **Phase C — Benchmark enrichi** (articles + rapprochements combinés).
- **Analyse Louvain** sur le graphe bipartite → clusters thématiques automatiques (source pour M2).
- **Résolution renumérotation 2016** → prérequis pour M5.

### Jalon superviseur 16 mai

**Premier scoreboard** B1 (LLM seul) vs B2 (LLM + RAG) sur les benchmarks CRFPA + Rapprochements.

### Questions ouvertes

- Collaboration LegMLAI pour étendre Les-Audits ?
- Contact Doctrine pour ground truth doctrinale (moyen terme) ?
- Accord formel Hector AI pour réutiliser M6 en cadre académique ?

---

## Slide 17 — Points de décision pour le superviseur

1. **Stratégie renumérotation 2016** : confirmer option (b) `recodified_as` avant B1, ou différer en v2 du graphe ?
2. **Priorité semaine 3** : lancer B1+B2 sur les 2 benchmarks *ou* construire M4 V4 (QCM revirement auto) en premier ?
3. **Extension du CRFPA** : intégrer les sessions 2022-2024 pour monter à ~100 Q, ou geler à 38 Q et passer à M2/Applied QA ?
4. **Timing du gel benchmark** ↔ premier scoreboard : avril ou mai ?

---

## Annexe A — Format canonique des entités du graphe

### Nœud Decision (JP)

```python
{
  type: "decision",
  pourvoi_norm: "10-87525",          # clé primaire
  id_judilibre: "61403186e27736d2287a7439",
  ecli: "ECLI:FR:CCASS:2011:CR01803",
  chamber: "Crim.",
  date: "2011-03-22",
  solution: "Rejet",
  publication: "Publié au Bulletin",
  is_benchmark_src: True
}
```

### Nœud Article (format [[Format-Fondement-Juridique]])

```python
{
  type: "article",
  pair_key: "code_civil:1240",       # clé primaire
  code_slug: "code_civil",
  article_num: "1240",
  n_citations_corpus: 1030
}
```

### Arêtes

- `rel="rapproche"` : Decision → Decision (Phase A)
- `rel="cite"` : Decision → Article (Phase B)

---

## Annexe B — Pipeline synoptique

```
Sources
├── CNB sujets (2024, 2025) ─────┐
├── Cap'Barreau grilles ─────────┤──► Triangulation (10 subagents Opus) ──► 38 Q CRFPA
├── IEJ Strasbourg copies ───────┘
└── Judilibre enrichie ─────────────► Scan streaming 3 passes ─────────────► 1 532 Q rapprochements
                                         └─► Index pourvoi→id ─────────────► Graphes Phase A
                                              └─► + code_article_pairs ───► Graphes Phase B
                                                      └─► Spring layout ─► 6 figures PNG

Résultat : 1 570 questions + 6 graphes + 6 figures.
```

---

## Annexe C — Carte des papiers mobilisés cette semaine

| Rôle | Papier | Contribution |
|---|---|---|
| Format rubrique | OpenAI HealthBench (2025) | Strates de poids différentiés |
| Format rubrique | Shi et al. PLawBench (2026) | Rubric-based evaluation |
| M1 socle | [[Alhajar-2025-Les-Audits-Affaires]] | 2 670 cas FR (non utilisé directement cette semaine, reste prévu) |
| M6 socle | [[Hector-AI-Benchmark-Interne]] | Pattern Judge-on-Judge + 7 dimensions |
| Métriques | [[Butler-Butler-2026-Legal-RAG-Bench]] (Isaacus) | gᵢ / rᵢ / cᵢ |
| Contexte RAG FR | [[Harvard-LIL-2024-Open-French-Law-RAG]] | Diagnostic du RAG vectoriel FR |
| Référence KG juridique | [[Belikov-Raoult-2025-KG-Cassation]] | Premier KG de la jurisprudence FR |
| Référence GraphRAG | [[Microsoft-2024-GraphRAG]] | Architecture référence |
| Référence KG + LLM | [[Guha-2026-KG-Assisted-LLM-Legal-Reasoning]] | Argumentaire KG > RAG pur |

---

## Annexe D — Documents Obsidian produits cette semaine

### Journaux
- `01-Projet/journal/2026-04-20.md` — MVP CRFPA livré
- `01-Projet/journal/2026-04-21.md` — rapprochements + graphes + viz

### Design
- `05-Technique/benchmark/Design-Rubrique-Hierarchisee.md`
- `05-Technique/benchmark/Design-Benchmark-Rapprochements.md`
- `05-Technique/benchmark/Design-Graphes-Phase-AB.md`
- `05-Technique/benchmark/Recap-MVP-2026-04-20.md`

### Concepts (mis à jour)
- `03-Concepts/Format-Fondement-Juridique.md` + §7 empirique

---

## Connexions

- [[Week-1-2026-04-13-Presentation-Superviseur]] — semaine 1 (état de l'art)
- [[2026-04-14]] — décision priorité benchmark + baselines
- [[2026-04-20]] · [[2026-04-21]] — journaux de la semaine
- [[Benchmark-KG-Juridique-FR-Design]] — design global 6 modules + 5 configs
- [[Design-Rubrique-Hierarchisee]] · [[Design-Benchmark-Rapprochements]] · [[Design-Graphes-Phase-AB]]
- [[Format-Fondement-Juridique]] · [[Format-Jurisprudence]]
