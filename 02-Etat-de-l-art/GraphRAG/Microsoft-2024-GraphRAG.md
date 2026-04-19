---
tags:
  - article
  - graphrag
categorie: GraphRAG
titre_complet: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
auteurs: "Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Larson"
annee: 2024
type: "Paper (arXiv) + projet open-source"
venue: "arXiv:2404.16130"
url: https://arxiv.org/pdf/2404.16130
doi: ""
pdf_local: ""
status: lu
pertinence: haute
created: 2026-04-09
modified: 2026-04-13
---

# Microsoft GraphRAG (Edge et al. 2024)

> [!info] Métadonnées
> **Auteurs** : Edge, Trinh, Cheng, Bradley, Chao, Mody, Truitt, Larson (Microsoft Research)
> **Année** : 2024 (v1 avril, v2 sept)
> **Type** : Paper de recherche + projet open-source
> **URL papier** : [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)
> **Repo** : [microsoft.github.io/graphrag](https://microsoft.github.io/graphrag/)

## Résumé

Papier fondateur de l'approche **GraphRAG** : pipeline complet pour répondre à des **global queries / sensemaking questions** sur de grands corpus (~1M tokens) via construction d'un KG par LLM, **détection hiérarchique de communautés (Leiden)**, **résumés de communautés**, et **inférence en map-reduce** sur ces résumés. Évalué en *Query-Focused Summarization* (QFS) avec un protocole **LLM-as-judge** sur 4 critères. Bat le RAG vectoriel naïf de **+20 à +30 points** en comprehensiveness/diversity, tout en consommant **97% de tokens en moins** que la baseline text summarization au niveau racine (C0).

## Contributions principales

1. **Pipeline de construction de KG par LLM** depuis du texte non structuré, avec extraction entités/relations (+ claims optionnels) et **gleanings** (passes successives auto-réflexives).
2. **Indexation hiérarchique par communautés Leiden** (niveaux C0 → Cn) avec **résumés JSON par communauté**, ancrés sur les IDs sources.
3. **Inférence map-reduce sur les community summaries** avec **scoring de helpfulness** et tri pour le reduce.
4. **Méthodologie d'évaluation QFS** : génération automatique de questions par personas + LLM-as-judge sur 4 critères (Comprehensiveness, Diversity, Empowerment, Directness).
5. **Argument empirique** : à corpus de ~1M tokens, GraphRAG bat le naive RAG sur les questions globales **avec un coût en tokens drastiquement réduit** (C0 = 2.6% du coût TS).
6. **Implémentation open-source** (Microsoft GraphRAG repo) — devenue référence d'architecture en 2024-2025.

---

## 1. Pipeline d'ensemble

> [!quote] Figure 1 du papier
> "Graph RAG pipeline using an LLM-derived graph index of source document text. This graph index spans nodes (e.g., entities), edges (e.g., relationships), and covariates (e.g., claims) that have been detected, extracted, and summarized by LLM prompts tailored to the domain of the dataset. Community detection (e.g., Leiden, Traag et al., 2019) is used to partition the graph index into groups of elements (nodes, edges, covariates) that the LLM can summarize in parallel at both indexing time and query time."

### Schéma textuel complet

```
[INDEXING TIME]
Source Documents
   | text extraction + chunking (600 tokens, overlap 100)
   v
Text Chunks
   | LLM extraction (prompt E.1) + gleanings (jusqu'à 3 passes)
   v
Element Instances  (entités, relations, optionnellement claims)
   | exact string matching + résumé LLM par nœud/arête
   v
Element Summaries  (poids d'arête = nb d'instances de la relation)
   | Leiden hiérarchique (graspologic)
   v
Graph Communities  (niveaux C0, C1, C2, C3)
   | prompt E.2 — priorisation par degré combiné des nœuds
   v
Community Summaries  (rapports JSON : title, summary, rating, findings)

[QUERY TIME]
User Query + Community Summaries (d'un niveau choisi)
   | shuffle + chunking
   | Map : prompt E.3 → réponse partielle + helpfulness [0-100]
   v
Community Answers  (triés par helpfulness, score 0 filtré)
   | Reduce : prompt E.4
   v
Global Answer
```

### Tableau récapitulatif des étapes

| Étape | Input | Transformation | Output | Modèle | Prompt |
|---|---|---|---|---|---|
| 1. Chunking | Documents bruts | Split fenêtres tokens | Chunks (600t, overlap 100) | Tokenizer | — |
| 2. Extraction | Chunks | Extraction + gleanings | Element Instances (tuples) | `gpt-4-turbo` | E.1 |
| 3. (Opt.) Claims | Chunks + entités | Extraction factuelle | Claims (covariates) | `gpt-4-turbo` | Claim Extraction |
| 4. Element Summaries | Instances multiples | Agrégation + résumé | Nœuds/arêtes résumés, poids = #instances | `gpt-4-turbo` | "domain-tailored" |
| 5. Communautés | KG pondéré | Leiden hiérarchique | Hiérarchie C0..Cn | Leiden (graspologic) | — |
| 6. Community Summaries | Communautés | Template "report-like" priorisé | JSON par communauté | `gpt-4-turbo` | E.2 |
| 7. Map (query) | Query + summaries | Réponse partielle + helpfulness | Answers scorés | `gpt-4-turbo` | E.3 |
| 8. Reduce (query) | Answers triés | Fusion contextuelle | Global Answer | `gpt-4-turbo` | E.4 |

---

## 2. Construction du graphe — détails techniques

### 2.1 Extraction d'entités et de relations (Element Instances)

> [!quote] §3.1.2
> "In this step, the LLM is prompted to extract instances of important *entities* and the *relationships* between the entities from a given chunk. Additionally, the LLM generates short descriptions for the entities and relationships."

**Format de sortie** : tuples délimités dans une liste unique :
```
("entity"<TD>FED<TD>ORGANIZATION<TD>The Fed is...)
<RD>("relationship"<TD>JEROME POWELL<TD>FED<TD>...<TD>9)
```
- Entité : `name`, `type`, `description`
- Relation : `source`, `target`, `description`, `relationship_strength` (score numérique)

#### Prompt E.1 (extrait verbatim)

> [!quote] Appendix E.1 — Element Instance Generation
> "---Goal--- Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
>
> ---Steps---
> 1. Identify all entities. For each identified entity, extract:
>    - entity_name (capitalized)
>    - entity_type: One of [{entity_types}]
>    - entity_description: Comprehensive description
>
> Format: ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
>
> 2. From the entities identified, identify all pairs of (source, target) that are *clearly related*. Extract:
>    - source_entity / target_entity / relationship_description / relationship_strength: numeric score
>
> Format: ("relationship"{tuple_delimiter}<source>{tuple_delimiter}<target>{tuple_delimiter}<description>{tuple_delimiter}<strength>)
>
> 3. Return as a single list. Use **{record_delimiter}** as list delimiter.
> 4. When finished, output {completion_delimiter}"

#### Customisation domaine

> [!quote] §3.1.2
> "These prompts can be tailored to the domain of the document corpus by choosing domain appropriate few-shot exemplars for in-context learning. [...] domains with specialized knowledge (e.g., science, medicine, **law**) will benefit from few-shot exemplars specialized to those domains."

> [!tip] Pour le KG juridique FR
> Remplacer `entity_types` par une ontologie juridique : `JURIDICTION, MAGISTRAT, ARTICLE_CODE, DECISION, PARTIE, INFRACTION, FONDEMENT_LEGAL, NOTION_JURIDIQUE`. Fournir 2-3 exemples annotés sur arrêts de Cassation. Traduire les prompts en français (`Return output in French`).

### 2.2 Gleanings (self-reflection)

> [!quote] Appendix A.2
> "Larger chunk size is less costly [...] However, the LLM tends to extract few entities from chunks of larger size. GPT-4 extracted almost twice as many entity references when the chunk size was 600 tokens than when it was 2400. To address this, we deploy a self-reflection prompt engineering approach. After entities are extracted from a chunk, we provide the extracted entities back to the LLM, prompting it to 'glean' any entities that it may have missed. We first ask the LLM to assess whether all entities were extracted, using a **logit bias of 100 to force a yes/no decision**. If the LLM responds that entities were missed, then a continuation indicating that 'MANY entities were missed in the last extraction' encourages the LLM to detect these missing entities."

- **Jusqu'à 3 itérations** (Figure 3).
- **Astuce technique** : `logit_bias = 100` sur tokens yes/no pour forcer une décision binaire avant continuation.
- **Trade-off** : chunk size + gleanings est un levier de coût/qualité — chunks de 600 sans gleaning ≈ chunks de 2400 avec 2 gleanings.

### 2.3 Element Summaries

> [!quote] §3.1.3
> "The use of an LLM to extract entities, relationships, and claims is already a form of abstractive summarization — these are meaningful summaries of concepts that, in the case of relationships and claims, may not be explicitly stated in the text. The entity/relationship/claim extraction processes creates multiple instances of a single element because an element is typically detected and extracted multiple times across documents. [...] Entity descriptions are aggregated and summarized for each node and edge. **Relationships are aggregated into graph edges, where the number of duplicates for a given relationship becomes edge weights.**"

#### Entity matching

> [!quote] §3.1.3
> "Our analysis uses **exact string matching** for *entity matching* [...] However, softer matching approaches can be used. GraphRAG is generally resilient to duplicate entities since duplicates are typically clustered together for summarization in subsequent steps."

> [!warning] Limite pour le juridique
> Exact string matching insuffisant : variantes d'écriture d'articles (`art. 1382` vs `article 1382 du Code civil`), alias de juridictions, numéros d'arrêts. Prévoir un étage d'**entity resolution** (embeddings + règles métier) en amont.

### 2.4 Détection de communautés — Leiden hiérarchique

> [!quote] §3.1.4
> "We use **Leiden community detection** (Traag et al., 2019) in a hierarchical manner, recursively detecting sub-communities within each detected community until reaching leaf communities that can no longer be partitioned. Each level of this hierarchy provides a community partition that covers the nodes of the graph in a **mutually-exclusive, collectively exhaustive** way, enabling divide-and-conquer global summarization."

**Pourquoi Leiden ?** Garantit des communautés bien connectées (Traag 2019) ; hiérarchique → MECE à chaque niveau.

**Implémentation** : `graspologic` (Chung et al. 2019).

#### Niveaux observés (Table 2)

| Dataset | C0 | C1 | C2 | C3 |
|---|---|---|---|---|
| Podcast (~1M tokens) | 34 | 367 | 969 | 1310 |
| News (~1.7M tokens) | 55 | 555 | 1797 | 2142 |

- **C0** = root-level (modularité maximum) — peu de communautés, très générales.
- **C3** = feuilles — nombreuses, fines.

### 2.5 Community Summaries

> [!quote] §3.1.5 — Priorisation
> "**Leaf-level communities.** Element summaries are prioritized and iteratively added to the LLM context window until token limit. Prioritization: **for each community edge in decreasing order of combined source and target node degree (i.e., overall prominence), add descriptions of the source node, target node, the edge itself, and related claims.**
>
> **Higher-level communities.** If all element summaries fit within the token limit, proceed as for leaf-level. Otherwise, rank sub-communities by element summary tokens (decreasing) and iteratively substitute sub-community summaries (shorter) for their associated element summaries (longer) until they fit."

#### Format JSON du rapport (prompt E.2)

> [!quote] Appendix E.2 — Community Summary Generation
> "---Goal--- Write a comprehensive report of a community, given a list of entities that belong to the community as well as their relationships and optional associated claims. The report will be used to inform decision-makers about information associated with the community and their potential impact.
>
> ---Report Structure---
> - TITLE
> - SUMMARY (executive)
> - IMPACT SEVERITY RATING (float 0-10)
> - RATING EXPLANATION
> - DETAILED FINDINGS (5-10 key insights)
>
> Return as JSON: {\"title\", \"summary\", \"rating\", \"rating_explanation\", \"findings\": [{\"summary\", \"explanation\"}, ...]}"

#### Grounding (traçabilité)

> [!quote] Appendix E.2 — Grounding Rules
> "Points supported by data should list their data references as follows: 'This is an example sentence supported by multiple data references [Data: <dataset name> (record ids); ...]' Do not list more than 5 record ids in a single reference. Instead, list the top 5 most relevant record ids and add '+more'. Do not include information where the supporting evidence for it is not provided."

> [!important] Pour le juridique
> Le grounding par IDs est précieux : on peut citer les arrêts source. Adapter le template E.2 — retirer `legal compliance / technical capabilities / reputation` (orienté business), conserver SUMMARY + FINDINGS + grounding.

---

## 3. Méthode de query (Map-Reduce)

> [!quote] §3.1.6
> "For a given community level, the global answer to any user query is generated as follows:
> - **Prepare community summaries.** Community summaries are randomly shuffled and divided into chunks of pre-specified token size. This ensures relevant information is distributed across chunks, rather than concentrated (and potentially lost) in a single context window.
> - **Map community answers.** Intermediate answers are generated in parallel. The LLM is also asked to generate a score between 0-100 indicating how helpful the generated answer is. Answers with score 0 are filtered out.
> - **Reduce to global answer.** Intermediate answers are sorted in descending order of helpfulness score and iteratively added into a new context window until the token limit is reached. This final context generates the global answer."

#### Helpfulness scoring (prompt E.4)

> [!quote] Appendix E.4
> "Generate an integer score between 0-100 that indicates how **helpful** this response is in answering the user's question. Return: `<ANSWER_HELPFULNESS> score_value </ANSWER_HELPFULNESS>`."

**Pourquoi shuffle avant chunking ?** Éviter que des summaries pertinents se regroupent dans un même chunk map et soient noyés (effet *lost in the middle*).

---

## 4. Hyperparamètres et coûts

| Paramètre | Valeur | Source |
|---|---|---|
| LLM (toutes étapes) | `gpt-4-turbo` | §4.1.3 |
| Chunk size (indexation) | **600 tokens** | §5.1 / §4.1.1 |
| Chunk overlap | **100 tokens** | §4.1.1 |
| Context window (génération) | **8k tokens** (fixe) | §4.1.3, App. C |
| Gleanings | jusqu'à **3 itérations** | Fig. 3 |
| Logit bias yes/no (gleaning) | **100** | App. A.2 |
| Algorithme communauté | **Leiden hiérarchique** (graspologic) | §3.1.4 |
| Niveaux évalués | C0, C1, C2, C3 | §4.1.2 |
| Recommandation qualité/coût | **C0 ou C1** | §5.1 |

### Choix du context window (App. C)

> [!quote] App. C
> "Surprisingly, the **smallest context window size tested (8k) was universally better** for all comparisons on comprehensiveness (average win rate of 58.1%). Given our preference for more comprehensive and diverse answers, we therefore used a fixed context window size of 8k tokens for the final evaluation."

→ Contre-intuitif : 16k/32k/64k dégradent la comprehensiveness (effet *lost in the middle* confirmé).

### Coût d'indexation observé

> [!quote] §4.1.3
> "Graph indexing with a 600 token window took **281 minutes for the Podcast dataset**, running on a virtual machine [...] using a public OpenAI endpoint for gpt-4-turbo (2M TPM, 10k RPM)."

### Coût à l'inférence — tokens par query (Table 2)

| Condition | Podcast | News | % du max |
|---|---|---|---|
| **C0** (root) | 26 657 | 39 770 | **2.3-2.6 %** |
| C1 | 225 756 | 352 641 | ~21 % |
| C2 | 565 720 | 980 898 | ~56 % |
| C3 | 746 100 | 1 140 266 | ~67-73 % |
| **TS** (text summarization baseline) | 1 014 611 | 1 707 694 | 100 % |

> [!important] Insight clé
> **C0 coûte ~40× moins cher que TS** tout en gardant 72% de winrate sur comprehensiveness. C'est l'argument économique majeur de GraphRAG.

### Tailles de graphes construits

| Dataset | Nodes | Edges |
|---|---|---|
| Podcast (~1M tokens) | 8 564 | 20 691 |
| News (~1.7M tokens) | 15 754 | 19 520 |

---

## 5. Évaluation

### 5.1 Tâche : Query-Focused Summarization (QFS)

> [!quote] §1
> "RAG fails on global questions directed at an entire text corpus, such as 'What are the main themes in the dataset?', since this is inherently a query-focused summarization (QFS) task, rather than an explicit retrieval task."

> [!quote] §2
> "[Vector RAG] works well for queries that can be answered with information localized within a small set of records. However, vector RAG approaches do not support **sensemaking queries**, meaning queries that require global understanding of the entire dataset."

**Distinction fondamentale** : *local retrieval* (QA factuel) vs *global sensemaking* (synthèse transversale). GraphRAG cible le second.

> [!tip] Transposition juridique
> "Quels sont les courants jurisprudentiels de la Cour de cassation sur X en 2020-2025 ?" est une question de sensemaking. Un KG juridique justifie sa valeur **précisément sur ce type de requête**, pas sur "Quelle est la date de l'arrêt X ?".

### 5.2 Datasets

| Corpus | Chunks | Tokens | Domaine |
|---|---|---|---|
| **Podcast transcripts** (*Behind the Tech*, K. Scott) | 1 669 | ~1.0 M | Conversations tech/science |
| **News articles** (benchmark 2013-2023) | 3 197 | ~1.7 M | Entertainment, business, sports, tech, health, science |

> [!warning] Limite reconnue
> "Our evaluation to date has focused on sensemaking questions specific to two corpora each containing approximately 1 million tokens. **More work is needed to understand how performance generalizes to datasets from various domains.**" → Aucune validation sur corpus juridique.

### 5.3 Génération automatique des questions (méthode persona)

Algorithme 1 — `K = N = M = 5` → **125 questions par dataset** :

> [!quote] App. E
> "Based on the corpus description, prompt the LLM to:
> 1. Describe personas of K potential users of the dataset.
> 2. For each user, identify N tasks relevant to the user.
> 3. Specific to each user & task pair, generate M high-level questions"

**Contrainte** : questions globales, pas factuelles.

**Exemple persona podcast** : "A tech journalist looking for insights and trends in the tech industry" (Table 1).

> [!tip] Méthode persona transposable au droit
> Personas : magistrat / avocat / chercheur en droit / justiciable / LegalTech.
> Tâches : préparer un pourvoi / rédiger doctrine / conseiller un client.
> Questions : "Quelles sont les tendances sur l'indemnisation du préjudice moral ?", "Comment les Chambres divergent sur la clause abusive ?".

### 5.4 Conditions comparées (6 conditions, contexte fixe 8k)

| Code | Nom | Description |
|---|---|---|
| **C0** | Root community summaries | Résumés des communautés racines |
| **C1** | High-level summaries | Sous-communautés de C0 |
| **C2** | Intermediate summaries | Sous-communautés de C1 |
| **C3** | Low-level summaries | Feuilles |
| **TS** | Text Summarization | Map-reduce sur **textes source** mélangés (pas de graphe) |
| **SS** | Semantic Search (= Naive RAG) | Vector RAG : top-k chunks par similarité |

> [!info] Naive RAG = SS dans le papier
> Le baseline RAG vectoriel est noté **SS** (Semantic Search), pas "Naive RAG". Fenêtre 8k identique partout pour comparabilité.

### 5.5 Métrique principale : LLM-as-judge

#### Protocole

- **Juge** : LLM (GPT-4 vraisemblable, non explicité comme juge).
- **Comparaison head-to-head par paires** : le juge reçoit la question + 2 réponses + un critère, et choisit.
- **125 questions × 5 répétitions** moyennées.

#### Les 4 critères (définitions verbatim, §3.3)

| Critère | Définition |
|---|---|
| **Comprehensiveness** | "How much detail does the answer provide to cover all aspects and details of the question?" |
| **Diversity** | "How varied and rich is the answer in providing different perspectives and insights on the question?" |
| **Empowerment** | "How well does the answer help the reader understand and make informed judgments about the topic?" |
| **Directness** (controle) | "How specifically and clearly does the answer address the question?" |

> [!info] Pourquoi Directness est un *control*
> Directness est attendu en **tension** avec Comprehensiveness/Diversity (réponse directe = courte = moins exhaustive). Sert de **sanity check** : si un système gagne aussi sur Directness, c'est suspect (biais du juge).

#### Gestion des biais

- **Biais d'ordre** : 5 répétitions moyennées (swap possible).
- **Biais de longueur** : non traité explicitement → **limite reconnue**.
- **Validation** : référencent Zheng et al. 2024 mais **pas d'étude d'accord inter-juge LLM vs humain** dans le papier.

### 5.6 Résultats (Figure 2)

| Dataset | Critère | GraphRAG (C0-C3) vs SS | TS vs SS |
|---|---|---|---|
| Podcast | Comprehensiveness | **72-83 %** (p<.001) | TS gagne aussi |
| Podcast | Diversity | **75-82 %** (p<.001) | TS gagne aussi |
| Podcast | Directness | SS gagne (controle) | SS gagne |
| News | Comprehensiveness | **72-80 %** (p<.001) | TS gagne aussi |
| News | Diversity | **62-71 %** (p<.01) | TS gagne aussi |
| News | Directness | SS gagne (controle) | SS gagne |

#### Verdicts

- **GraphRAG ≫ Naive RAG (SS)** : marge de **+20 à +30 points** en Comprehensiveness/Diversity.
- **GraphRAG ≈ TS** en qualité, mais **TS coûte 40-100× plus cher** en tokens.
- **C0 = sweet spot** :
  > "Root-level GraphRAG offers a highly efficient method for the iterative question answering that characterizes sensemaking activity, while retaining advantages in comprehensiveness (72% win rate) and diversity (62% win rate)."
- **Directness** : SS gagne partout → confirme l'absence de biais général pro-GraphRAG du juge.

---

## 6. Discussion et limites

### Limites reconnues par les auteurs

- "More work is needed to understand how performance generalizes to datasets from various domains."
- "Comparison of fabrication rates [...] would also strengthen the current analysis." → **Pas de mesure de factualité dans l'évaluation principale**.

### Validité du LLM-as-judge

| Forces | Faiblesses |
|---|---|
| 125 × 5 répétitions = N statistiquement sain | Pas d'accord inter-juge LLM vs humain |
| Critère de contrôle (Directness) fonctionne | Pas de traitement du biais de longueur |
| Méthodologie répliquable | Même famille de modèle pour générer ET juger (risque self-preference) |

### Reproductibilité

- Datasets : podcasts publics, news = benchmark commercial (partiel).
- Prompts : appendices E (system) et F (eval) du PDF.
- Code : repo Microsoft GraphRAG open-source.
- **Coût de réplication** : 125 × 5 × 6 conditions × 2 datasets = ~7 500 appels juge + toute la génération en amont.

---

## 7. Synthèse pour notre projet KG juridique français

### 7.1 Ce qui se transpose tel quel

| Brique Microsoft | Réutilisable pour le droit |
|---|---|
| Pipeline 8 étapes (chunking → extraction → summaries → communautés → query map-reduce) | **Oui**, squelette directement adoptable |
| Prompt E.1 (extraction entités/relations avec types paramétrables) | **Oui**, à reparamétrer avec ontologie juridique |
| Gleanings (logit bias + continuation) | **Oui**, particulièrement utile sur arrêts longs et denses |
| Leiden hiérarchique sur graphe entité-entité pondéré | **Oui à tester** : les communautés ont-elles du sens en jurisprudence (par matière ? par chambre ? par notion ?) |
| Format JSON des community summaries avec grounding par IDs | **Oui**, le grounding est essentiel en droit (citer l'arrêt source) |
| Map-reduce avec helpfulness scoring | **Oui**, simple et efficace |
| 4 critères Comprehensiveness / Diversity / Empowerment / Directness | Oui, **3 sur 4** (Directness comme contrôle uniquement) |
| Méthode persona-based pour générer les questions | **Oui**, parfaitement adaptable (juriste/avocat/chercheur/justiciable/LegalTech) |
| Conditions comparatives C0-C3 vs SS vs TS | **Oui**, plus baselines BM25 et Judilibre+LLM |

### 7.2 Ce qui doit être adapté

| Élément | Adaptation requise |
|---|---|
| `entity_types` génériques | Ontologie juridique (`JURIDICTION, MAGISTRAT, ARTICLE_CODE, DECISION, PARTIE, INFRACTION, FONDEMENT_LEGAL, NOTION_JURIDIQUE`) |
| Few-shot exemplars | 2-3 exemples annotés sur arrêts Cassation (style "Vu l'article... ; attendu que...") |
| Langue de sortie | `Return output in French` partout (E.1, E.2, E.3, E.4) |
| Template community summary E.2 | Retirer "legal compliance / technical capabilities / reputation" (orienté business) |
| Exact string matching | Remplacer par entity resolution (embeddings + règles métier : alias d'articles, dates, numéros) |
| Pas de mesure de factualité | **Ajouter critères propres au juridique** (cf. ci-dessous) |

### 7.3 Critères d'évaluation à ajouter pour le juridique

> [!warning] Manques critiques chez Edge et al. pour usage juridique
> Le papier n'inclut **aucune métrique de factualité ni de traçabilité**. En droit, c'est rédhibitoire.

À ajouter :
- **Factualité de la citation** : arrêt cité existe-t-il ? date/numéro/chambre exacts ? (vérification automatique vs Judilibre).
- **Traçabilité / grounding** : chaque assertion reliée à un passage source ? (Claimify-style + lien texte).
- **Conformité à la doctrine dominante** : reflète-t-elle la solution majoritaire ou hallucine-t-elle une position minoritaire ?
- **Hiérarchie des sources** : Cassation > Cour d'appel, texte > coutume, etc.
- **Absence de conseil juridique hallucination** (responsabilité potentielle).

### 7.4 Stratégie d'évaluation MVP

> [!tip] Plan d'évaluation pragmatique
> 1. **1 dataset** : toutes les décisions Cass. civ. 1re 2020-2024 sur un thème précis (~500 arrêts).
> 2. **K=N=M=3** → 27 questions (au lieu de 125) — réduit les coûts.
> 3. **3 répétitions** (au lieu de 5).
> 4. **3 conditions** : C0 + C2 + SS (Naive RAG) — éviter C3 et TS au début (trop chers).
> 5. **5 critères** : Comprehensiveness, Diversity, Empowerment + Factualité-citation + Grounding.
> 6. Valider le protocole avant scaling à K=N=M=5 et à plusieurs chambres/matières.

### 7.5 Risques anticipés

> [!warning] Points de vigilance pour le portage juridique
> 1. **Coût d'indexation** : 281 min pour 1M tokens podcast ; la jurisprudence Cassation = des dizaines de millions de tokens → budget plusieurs dizaines d'heures GPU/API + plusieurs k€.
> 2. **Hallucinations sur noms/numéros d'arrêts** : risque majeur ; collaborative refinement (cf. Zhang 2025) non-négociable.
> 3. **Évolution du droit** : nécessite KG incrémental ; exclut le pure fine-tuning.
> 4. **Communautés Leiden sur jurisprudence** : à valider empiriquement — les clusters obtenus correspondent-ils à des matières juridiques cohérentes ou à du bruit ?
> 5. **Sensibilité au format de prompt** (cf. Zhang 2025) : les community summaries français doivent respecter les conventions juridiques.

---

## 8. Connexions

### Articles liés
- [[Zhang-2025-GraphRAG-Survey-Customized-LLMs]] — survey qui place GraphRAG comme architecture de référence du paradigme *Hybrid GraphRAG*.
- [[Peng-2024-GraphRAG-Survey-ACM]] — première survey systématique GraphRAG.
- [[Han-2025-RAG-With-Graphs-Survey]] — autre survey 2025.
- [[Belikov-Raoult-2025-KG-Cassation]] — application juridique française à comparer.
- [[DAmato-2025-KG-Violence-Women-CEDH]] — autre application légale.

### Concepts liés
- [[Leiden community detection]]
- [[Community summary]]
- [[Map-reduce inference]]
- [[Gleanings - self-reflection]]
- [[Hybrid Graph]]
- [[LLM-as-judge]]
- [[Query-Focused Summarization]]
- [[Sensemaking vs Retrieval]]

### Questions soulevées
- Les communautés Leiden ont-elles du sens sur de la jurisprudence ? Quelle granularité ?
- Quel niveau (C0/C1/C2) optimise qualité/coût pour le droit ?
- Comment intégrer l'ontologie juridique existante (LYNX, LKIF) dans le prompt E.1 ?
- Faut-il un schéma fixe ou laisser le LLM extraire librement ?
- Comment évaluer la **factualité** d'une citation d'arrêt automatiquement ?

---

## 9. Citations clés

> "RAG fails on global questions directed at an entire text corpus [...] this is inherently a query-focused summarization task, rather than an explicit retrieval task." (§1)

> "Root-level GraphRAG offers a highly efficient method for the iterative question answering that characterizes sensemaking activity, while retaining advantages in comprehensiveness (72% win rate) and diversity (62% win rate) over vector RAG." (§5.1)

> "Domains with specialized knowledge (e.g., science, medicine, law) will benefit from few-shot exemplars specialized to those domains." (§3.1.2)

> "The smallest context window size tested (8k) was universally better for all comparisons on comprehensiveness." (App. C)

---

## 10. Notes personnelles

- **Référence d'architecture** incontournable du champ. Tout système GraphRAG postérieur s'y compare ou s'en inspire (LightRAG, KAG, HippoRAG, HYBGRAG).
- L'argument **C0 = 2.6% du coût de TS** est l'**argument économique central** à retenir pour défendre l'approche en mémoire.
- La méthode **persona-based de génération de questions** est probablement la contribution méthodologique la plus sous-estimée du papier — directement réutilisable pour tout nouveau benchmark.
- **Faiblesse majeure** pour notre cas : pas de mesure de factualité. C'est **précisément là** qu'une contribution juridique peut innover (claim-grounding sur arrêts vérifiables via Judilibre).
- Récupérer les **prompts complets** des appendices E et F directement depuis le PDF avant toute réimplémentation (les agents n'ont pas pu reproduire 100% du verbatim).
- Le repo Microsoft est en évolution rapide (LazyGraphRAG depuis nov 2024) — vérifier la dernière version pour les dernières optimisations.
