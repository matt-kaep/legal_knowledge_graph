---
tags: [benchmark, methodologie, design, technique]
type: design-document
status: "en-cours"
created: 2026-04-16
modified: 2026-04-20
---

> [!info] MAJ 2026-04-20 — Format d'évaluation précisé
> Le format de ground truth et les métriques ont été précisés dans [[Design-Rubrique-Hierarchisee]] :
> - Rubrique hiérarchisée **core / expected / expert** (pondérée 3/2/1)
> - Sortie JSON imposée aux systèmes testés (schéma unique)
> - Formats canoniques fixés pour les sources : [[Format-Fondement-Juridique]] (articles/codes) et [[Format-Jurisprudence]] (JP)
> - Abandon de l'étiquette qualitative `Difficulty` pour M1-M5 au profit de proxies mesurables (`|rubric|`, `nb_sources_attendues`, `ratio_expert`)
> - Détails et motivation : [[2026-04-20]]

> [!important] MAJ 2026-04-21 (soir) — Fusion vers un benchmark unifié "mode avocat"
> Décision actée avec Jhony : les deux benchmarks construits le 20 et 21 avril
> (CRFPA structuré + Rapprochements) sont fusionnés en **un seul benchmark
> "mode avocat"** qui correspond au besoin métier réel.
> - Spec complète : [[Design-Benchmark-Avocat-Unifie]]
> - Input : question juridique en langage avocat.
> - Output : `{articles, JP avec sens favorable/défavorable, arguments}`.
> - Baselines : B1 (LLM) · B2 (LLM+RAG) · C1 (LLM+GraphRAG) · C2 (GNN+BERT) · C3 (retrieval pur).
> - Scoring : `w_art·F1_pénalisé + w_jp·F1_pondéré_sens + w_arg·similarité_sémantique`.
> - Conséquence sur ce document : les modules M1-M6 restent l'ossature *conceptuelle*
>   mais le benchmark *exécutable* est désormais unifié.

# Design du benchmark KG juridique FR

> Document de conception du benchmark consolidé, synthétisant tous les inputs de l'état de l'art et de la réunion superviseur du [[2026-04-14]].

---

## 1. Objectifs

### Objectif général
Construire **un benchmark complet** pour évaluer la pertinence d'un système de recherche juridique français, couvrant :
- les **LLMs seuls** (connaissance intrinsèque)
- les **pipelines RAG** classiques
- les **pipelines agentiques / multi-step**
- notre **système cible** : GraphRAG sur KG juridique

### Objectif métier final
> *"Pour une question juridique donnée, retrouver les bons articles et la bonne jurisprudence et produire une réponse justifiée, traçable et juste."*

### Ce que le benchmark doit mesurer
1. **Justesse** de la réponse (exactitude factuelle)
2. **Traçabilité** (capacité à citer les vraies sources)
3. **Couverture** (articles + JP pertinents remontés)
4. **Raisonnement multi-saut** (article → JP → article cité → JP interprétative)
5. **Sensibilité** (robustesse aux reformulations)
6. **Reproductibilité** (variance entre runs)
7. **Temporalité** (droit à une date donnée)

---

## 2. Architecture du benchmark — 6 modules

```
┌───────────────────────────────────────────────────────────────────┐
│  BENCHMARK KG JURIDIQUE FR                                        │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐            │
│  │  Module 1   │  │  Module 2   │  │   Module 6 ⭐   │            │
│  │ Principe QA │  │ Applied QA  │  │  Interprétation │            │
│  │ (Les-Audits)│  │ (ouverte)   │  │  d'arrêt        │            │
│  │   Niveau 1  │  │   Niveau 2  │  │  contextualisée │            │
│  └─────────────┘  └─────────────┘  └─────────────────┘            │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │  Module 3   │  │  Module 4   │  │  Module 5   │                │
│  │  Multi-saut │  │  QCM        │  │  Temporalité│                │
│  │  (KG)       │  │  retrieval  │  │  (versioning│                │
│  │             │  │             │  │   temporel) │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
└───────────────────────────────────────────────────────────────────┘
```

### Deux niveaux de raisonnement juridique

Le benchmark distingue deux niveaux fondamentaux :

| Niveau | Question type | Output attendu | Modules concernés |
|---|---|---|---|
| **Niveau 1 — Principe** | *"Quelles sont les conditions de X ?"* | Articles + JP de principe (CC) + interprétations (CA) | M1, M3, M4, M5 |
| **Niveau 2 — Application** | *"Dans mon dossier, X est-il valable ?"* | Articles + JP principe + **JP d'espèce** (TJ/TC/CA similaires) | M2, M6 |

La JP est taxonomisée hiérarchiquement selon le champ `rank` :
- **principe** → Cassation (CC) — pose la règle
- **interprétation** → Cour d'appel (CA) — précise
- **espèce** → TJ / TC / premières instances — applique aux faits

---

### Module 1 — Principe QA (niveau 1, sans contexte)
- **Source** : [[Alhajar-2025-Les-Audits-Affaires]] (LegMLAI) + questions abstraites de [[Hector-AI-Benchmark-Interne]]
- **Format** : question générale → articles + JP principe + réponse
- **Contexte** : aucun
- **Taille cible** : 300-500 cas
- **Statut** : ✅ **base existante réutilisable**
- **Apport de notre benchmark** : tester les 5 configurations dessus + ajouter la dimension JP

### Module 2 — Applied QA (niveau 2, avec contexte, ouverte)
- **Source** : cases Hector synthétiques + construction manuelle
- **Format** : question + case_summary + client_position → articles + JP principe + **JP d'espèce** + réponse contextualisée
- **Taille cible** : 100-200 cas
- **Statut** : 🔨 **à construire**
- **Particularité** : mesure la *qualification des faits* et le *retrieval de JP d'espèce*
- **Méthode** : stratification 40% facile / 40% moyen / 20% difficile

### Module 6 — Interprétation d'arrêt contextualisée ⭐ (niveau 2, JP fournie)
- **Source** : cases Hector (analyse gold standard à produire) + construction manuelle fresh
- **Format** : question + case_summary + client_position + **1 décision fournie** → analyse à 7 dimensions
- **Taille cible** : 150-200 cas
- **Statut** : 🔨 **à construire** (pipeline de double-judge inspiré de Hector)
- **Originalité** : aucun benchmark FR ne teste *l'interprétation contextuelle d'une décision donnée*
- **Tâche distincte** du retrieval (la décision est fournie) et de la QA ouverte (contexte client précis)

**7 dimensions évaluées pour chaque paire (dossier, décision)** :

| Dimension | Description | Métrique |
|---|---|---|
| 1. `camp_in_decision` | Qui représente la position du client dans cette décision ? | binaire |
| 2. `sens_arret` | Cassation / rejet / confirmation / infirmation | classification |
| 3. `is_favorable` | Décision favorable au client actuel ? | **binaire — critique** |
| 4. `dispositif_summary` | Résumé de ce que décide la cour | similarité sémantique + LLM-judge |
| 5. `relevance` | Degré de pertinence vis-à-vis du dossier | MAE vs score humain (0-1) |
| 6. `principles_extracted` | Principes de droit extraits, utilisables | F1 sur liste de référence |
| 7. `transfer_reasoning` | Justification du transfert au dossier actuel | LLM-judge + validation humaine |

**Stratification** :
- 40% favorable / 40% défavorable / 20% ambigu ou pièges
- 40% facile / 40% moyen / 20% difficile
- 5 spécialisations équilibrées (~30-40 cas chacune)

**Cas pièges spécifiques** :
- Décision qui *semble* favorable mais ne l'est pas (ex : cassation sur moyen procédural)
- Cassation partielle avec portée réelle limitée
- Faits apparemment proches mais avec distinction factuelle décisive

### Module 3 — Raisonnement multi-saut
- **Format** : question qui nécessite de chaîner plusieurs sauts (article → arrêt → article cité → JP interprétative → revirement)
- **Taille cible** : 50-100 cas
- **Source** : construites à partir du graphe de citations du KG
- **Statut** : 🔨 **à construire une fois le KG v1 opérationnel**
- **Particularité** : c'est ici qu'on attend le plus gros gain GraphRAG vs RAG

### Module 4 — QCM retrieval (inspiré CaseHOLD)
- **Source** : [[Format-QCM-benchmark-juridique-FR]]
- **4 variantes** :
  - V1 : article applicable parmi 5
  - V2 : JP de principe parmi 5
  - V3 : JP d'espèce factuellement proche parmi 5
  - V4 : détection de revirement parmi 5
- **Taille cible** : 500-1000 questions (génération auto)
- **Statut** : 🔨 **à construire** (génération automatique depuis KG v1)
- **Avantage** : évaluation automatique simple (accuracy)

### Module 5 — Temporalité
- **Format** : *"Quelle était la règle applicable le JJ/MM/AAAA ?"* ou *"L'arrêt X rendu sous l'article Y (désormais modifié) fait-il encore autorité ?"*
- **Taille cible** : 50-100 questions
- **Source** : articles abrogés + revirements identifiés
- **Statut** : 🔨 **à construire** (nécessite KG avec versioning temporel)
- **Originalité** : gap identifié dans toute la littérature — aucun benchmark FR ne teste ça

---

## 3. Les 5 configurations à benchmarker

Ordre croissant de sophistication, chacune testée sur les 5 modules.

| # | Configuration | Rôle | Ce qu'on mesure |
|---|---|---|---|
| **B1** | LLM seul (zero-shot) | baseline minimale | connaissance intrinsèque |
| **B2** | LLM + RAG vectoriel | RAG classique | apport retrieval vectoriel |
| **B3** | LLM multi-step / agentique (sans RAG) | raisonnement itératif | apport chaîne de pensée |
| **B4** | LLM + RAG + agentique | état de l'art "classique" | combinaison RAG + agent |
| **🎯 C** | LLM + **GraphRAG** (notre système) | cible | apport du graphe structuré |

### Variante transversale : gros LLM vs petit LLM + GraphRAG

Pour chaque configuration, **doubler l'expérience** :
- Version "gros LLM" (Opus, GPT-4o, Gemini 3)
- Version "petit LLM + GraphRAG" (Mistral 7B, Gemma 3, Saul-Instruct)

> Objectif : prouver qu'un **petit LLM + graphe** peut rivaliser avec un **gros LLM seul**.

---

## 4. Les modèles à tester

### LLMs open-source
| Modèle | Taille | Intérêt particulier |
|---|---|---|
| [[Colombo-2024-Saul-Instruct]] | 7B | fine-tuné juridique européen |
| Mistral-7B-Instruct-v0.3 | 7B | base naturelle FR |
| Mistral-7B + fine-tune [[VinceGx33-2025-Mistral-Legal-French-Dataset]] | 7B | tester le gain du fine-tuning FR |
| Gemma 3 | 7B / 27B | concurrent récent |
| Llama 3.x | 8B / 70B | référence anglo-saxonne |
| Qwen 2.5 | 7B / 72B | concurrent asiatique |
| MiniMax / Kimi | var. | optionnel |

### LLMs fermés (référence)
| Modèle | Rôle |
|---|---|
| GPT-4o / GPT-5.2 | baseline commerciale |
| Claude Opus 4.6 | baseline commerciale |
| Gemini 3.1 Pro | baseline commerciale |

### Embeddings à comparer (pour le RAG)
| Embedder | Notes |
|---|---|
| `intfloat/multilingual-e5-large` | utilisé par [[Harvard-LIL-2024-Open-French-Law-RAG]] |
| `BAAI/bge-m3` | multilingue performant |
| `Kanon 2 Embedder` | spécialisé juridique (anglais cependant) |
| OpenAI text-embedding-3-large | baseline commerciale |
| Custom fine-tuné FR juridique | piste de contribution |

---

## 5. Les métriques

Consolidation de ce qui a été vu dans la littérature + demandes superviseur.

### 5.1 Métriques d'exactitude (inspirées de Harvard LIL + Les-Audits)

| Métrique | Définition | Échelle |
|---|---|---|
| **Correctness** | réponse factuellement juste | binaire 0/1 |
| **Partial correctness** | réponse partiellement juste | 0 / 0.5 / 1 |
| **Coverage** | dimensions correctement couvertes (Les-Audits 5 dim) | 0 à 5 |

### 5.2 Taxonomie d'erreurs (inspirée Isaacus)

Pour chaque question, **3 métriques binaires mutuellement exclusives** :

| Métrique | Définition |
|---|---|
| **Groundedness** (gᵢ) | réponse soutenue par les sources récupérées |
| **Retrieval accuracy** (rᵢ) | passage pertinent récupéré |
| **Correctness** (cᵢ) | réponse correcte |

→ Permet de **décomposer les erreurs** :
- **Hallucination** (gᵢ=0) : réponse non soutenue → fabrication
- **Retrieval Error** (gᵢ=1, cᵢ=0, rᵢ=0) : le bon passage n'a pas été récupéré
- **Reasoning Error** (gᵢ=1, cᵢ=0, rᵢ=1) : le bon passage était là, raisonnement raté

### 5.3 Métriques de traçabilité (originales)

| Métrique | Définition |
|---|---|
| **Citation precision** | % des sources citées qui sont effectivement pertinentes |
| **Citation recall** | % des sources pertinentes qui sont citées |
| **Citation correctness** | citation exacte (n° article correct, référence d'arrêt correcte) |

### 5.4 Métriques de robustesse (demandes superviseur)

| Métrique | Définition | Méthode |
|---|---|---|
| **Sensibilité** | variance de la réponse face à des reformulations | 5 reformulations / question |
| **Reproductibilité** | variance entre runs identiques | 3-5 runs / question |
| **Reproductibilité isolée** | variance du retrieval seul vs génération seule | ablations |

### 5.5 LLM-as-judge — pattern Judge-on-Judge (inspiré Hector)

**Évaluation à 3 niveaux** — élimine le biais du juge unique, particulièrement critique sur les jugements subjectifs (`is_favorable`, qualité du raisonnement).

| Niveau | Rôle | Modèle |
|---|---|---|
| **Judge 1** | Évalue automatiquement la réponse du modèle testé | LLM fort (Grok Reasoning ou GPT-5) |
| **Judge 2 / meta-judge** | Relit la décision/sources et vérifie le verdict du Judge 1 | LLM différent (Claude Sonnet 4.6) |
| **Judge 3 / humain** | Validation sur échantillon | moi + juriste partenaire si possible |

Critères (inspirés de Microsoft GraphRAG + Hector) :
- Comprehensiveness
- Diversity
- Empowerment
- Directness
- **Correctness** (Hector-style : dispositif_ok, favorable_ok, principles_ok)

> Couverture humaine ciblée : 10-20% des cases, stratifiés par difficulté et par module.

### 5.6 Métriques spécifiques Module 6 (Interprétation d'arrêt)

Pour la tâche d'interprétation contextualisée, métriques dédiées :

| Métrique | Type | Méthode |
|---|---|---|
| Camp identifié | binaire | comparaison directe |
| Sens de l'arrêt | classification 4-classes | accuracy |
| is_favorable | binaire | accuracy **(métrique cœur)** |
| Dispositif summary | texte | similarité sémantique + LLM-judge |
| Relevance | régression [0,1] | MAE vs humain |
| Principles extracted | liste | F1 par matching (avec normalisation) |
| Transfer reasoning | texte libre | LLM-judge + humain sur échantillon |

### 5.7 Ground truth manuel

- Source : **Doctrine** (références doctrinales de référence) + **JP de rapprochement** (arrêts cités ensemble par les juristes) + **annotations Hector** (cases synthétiques)
- Construction : annotation manuelle d'un sous-ensemble (50-100 questions par module critique)
- Usage : référence "or" pour calibrer les autres métriques

---

## 6. Protocole expérimental

### 6.1 Stratification des questions

Pour chaque module, échantillon **stratifié** :
- 40% **facile** (1-saut, citation explicite, vocabulaire clair)
- 40% **moyen** (2-sauts, vocabulaire technique)
- 20% **difficile** (3+ sauts, revirements, motivation implicite)

> Évite le piège signalé par [[Harvard-LIL-2024-Open-French-Law-RAG]] et [[Zheng-2021-CaseHOLD]] : benchmarks trop faciles où tout le monde plafonne.

### 6.2 Conditions de test

- **Température** : 0.0 (déterministe) pour reproductibilité
- **Seed** : fixé
- **Prompts** : standardisés et versionnés (dans le repo)
- **5 runs** par question pour mesurer la reproductibilité
- **5 reformulations** par question (par LLM externe) pour mesurer la sensibilité

### 6.3 Anti-contamination

Inspiré de [[Alhajar-2025-Les-Audits-Affaires]] :
- Régénération périodique des questions depuis le corpus
- Cross-model : un modèle génère, un autre évalue
- Mise à jour depuis Légifrance en temps réel
- Variation personas / scénarios

---

## 7. Plan de construction

### Phase B0 — Setup (semaine 1)
- [ ] Télécharger Les-Audits-Affaires
- [ ] Cloner leur harness d'évaluation
- [ ] Setup cluster GPU (contact superviseur)
- [ ] Dépôt Git pour le benchmark

### Phase B1 — Baselines sur module 1 (semaines 2-4)
- [ ] Implémenter B1 (LLM seul) sur tous les modèles
- [ ] Implémenter B2 (LLM + RAG vectoriel) avec `multilingual-e5-large`
- [ ] Implémenter B3 et B4 (agentiques)
- [ ] Premier scoreboard sur Les-Audits-Affaires

### Phase B2 — Construction modules 2 et 6 (semaines 3-7)
- [ ] Extraire 500+ arrêts variés depuis Judilibre (CC, CA, TJ, TC)
- [ ] Pipeline d'extraction semi-auto des questions ouvertes (M2)
- [ ] Validation manuelle M2 (objectif : 100-200 questions validées)
- [ ] **Construction M6** : paires (dossier, décision) avec analyse gold standard à 7 dimensions
- [ ] Pipeline Judge-on-Judge opérationnel pour M6

### Phase B3 — Construction modules 3-5 (semaines 5-10)
- [ ] KG v1 opérationnel (citations brutes)
- [ ] Module 3 (multi-saut) généré depuis le KG
- [ ] Module 4 (QCM) généré automatiquement (4 variantes)
- [ ] Module 5 (temporalité) : versioning du KG + questions

### Phase B4 — Tests de la cible GraphRAG (semaines 10-14)
- [ ] Configuration C (LLM + GraphRAG)
- [ ] Tests sur tous les modules (incluant M6 qui est le plus critique)
- [ ] Comparaison finale vs B1-B4

### Phase B5 — Analyse et rédaction (semaines 14-16)
- [ ] Analyse statistique des résultats
- [ ] Rédaction de la section évaluation du mémoire
- [ ] Publication du benchmark sur HuggingFace

---

## 8. Outils et infrastructure

### Existants réutilisés
- `legmlai/les-audits-affaires` + `les-audits-evaluation-harness`
- `maastrichtlawtech/bsard` + `maastrichtlawtech/legal-camembert`
- `VinceGx33/mistral-legal-french-dataset` (fine-tuning)
- `louisbrulenaudet/legalkit` + `louisbrulenaudet/legifrance`
- Judilibre API + Jurica
- Saul-Instruct (HF : `Equall/Saul-Instruct-v1`)
- [[Hector-AI-Benchmark-Interne]] — 90+ questions synthétiques + pipeline Judge-on-Judge (sous réserve d'accord)

### À construire
- Pipeline d'extraction question/réponse depuis arrêts (M2)
- Pipeline de construction paires (dossier, décision) + gold analysis (M6)
- Harness d'évaluation étendu (6 modules)
- Générateur de QCM depuis le KG (M4)
- Versioning temporel des articles (M5)
- Dashboard de résultats (W&B ou MLflow)
- Pipeline Judge-on-Judge à 3 niveaux

### Inspirations méthodologiques
- [[Butler-Butler-2026-Legal-RAG-Bench]] — taxonomie d'erreurs (gᵢ/rᵢ/cᵢ)
- [[Alhajar-2025-Les-Audits-Affaires]] — évaluation 5 dimensions + personas
- [[Guha-2023-LegalBench]] — framework IRAC adaptable
- [[Zheng-2021-CaseHOLD]] — format QCM
- [[Harvard-LIL-2024-Open-French-Law-RAG]] — grille critique
- [[Hector-AI-Benchmark-Interne]] — **pattern Judge-on-Judge + 7 dimensions d'interprétation d'arrêt**
- [[Louis-2022-BSARD]] — retrieval statutaire FR + schéma hiérarchique d'articles

---

## 9. Livrables et publication

### Livrables
1. **Dataset public** sur HuggingFace : `FrenchLegalBench-Extended` (hypothèse de nom)
2. **Harness d'évaluation** open-source sur GitHub
3. **Rapport technique** (section du mémoire)
4. **Leaderboard public** (W&B ou site dédié)

### Cibles de publication
- Workshop NLP juridique (JURIX, ICAIL)
- Conférence NLP (EMNLP, ACL)
- Blog HF + communication avec la communauté FR juridique

---

## 10. Questions ouvertes

- [ ] Quelle **licence** pour notre benchmark ? (CC-BY-4.0 recommandé pour cohérence avec LegalBench)
- [ ] Où obtenir du **ground truth doctrinal** à grande échelle ? (Contact Doctrine ? Dalloz ?)
- [ ] Quel **modèle juge** utiliser pour le LLM-as-judge ? (doit être absent des candidats testés)
- [ ] Peut-on **collaborer avec LegMLAI** pour étendre conjointement Les-Audits-Affaires ?
- [ ] **Ordre de priorité** des modules si contrainte de temps ? (proposition : **1 → 6 → 2 → 4 → 3 → 5**)
- [ ] Accord explicite Hector AI pour réutiliser les questions synthétiques dans un cadre académique
- [ ] Identifier la JP d'espèce en ground truth : annotation manuelle vs embedding + validation échantillon

---

## Annexe — Carte des benchmarks étudiés

| Benchmark | Ce qu'on en retient |
|---|---|
| [[Harvard-LIL-2024-Open-French-Law-RAG]] | Diagnostic du problème du RAG vectoriel sur FR |
| [[Guha-2023-LegalBench]] | Framework IRAC ; échelle 162 tâches |
| [[Niklaus-2023-LEXTREME]] | Benchmark multilingue (FR limité à EUR-LEX + Suisse) |
| [[Zheng-2021-CaseHOLD]] | Format QCM transposable |
| [[Chalkidis-2022-LexGLUE]] | Structure benchmark multi-tâches |
| [[Alhajar-2025-Les-Audits-Affaires]] | **Base de notre benchmark** (M1) |
| [[Louis-2022-BSARD]] | Retrieval statutaire FR + schéma hiérarchique |
| [[Butler-Butler-2026-Legal-RAG-Bench]] | Taxonomie d'erreurs RAG |
| [[Hector-AI-Benchmark-Interne]] | **Base de M6 — interprétation d'arrêt + Judge-on-Judge** |
| LexEval (Li et al. 2024) | Chinois — inspiration structure multi-tâches |
| PLawBench (Shi et al. 2026) | Chinois — rubric-based evaluation |

---

## Prochaine entrée journal

→ [[2026-04-16]] : démarrage Phase B0 (setup)
