---
tags: [article, graphrag]
categorie: "GraphRAG"
titre_complet: "A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models"
auteurs: "Zhang, Chen, Bei, Yuan, Zhou, Hong, Chen, Xiao, Zhou, Dong, Chang, Huang"
annee: 2025
type: "Survey (arXiv)"
venue: "arXiv"
url: "arxiv.org/abs/2501.13958"
doi: ""
pdf_local: ""
status: "lu"
pertinence: "haute"
created: 2026-04-10
modified: 2026-04-13
---

# A Survey of GraphRAG for Customized LLMs (Zhang et al. 2025)

> [!info] Métadonnées
> **Auteurs** : Zhang, Chen, Bei, Yuan, Zhou, Hong, Chen, Xiao, Zhou, Dong, Chang, Huang (DEEP-PolyU)
> **Année** : 2025 (v1 jan 2025, v3 sept 2025)
> **Type** : Survey (26 pages)
> **URL** : [arxiv.org/abs/2501.13958](https://arxiv.org/abs/2501.13958)
> **Repo associé** : [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG)

## Résumé

Survey de référence (DEEP-PolyU, équipe qui maintient *Awesome-GraphRAG*) proposant un **framework formel en 3 phases** pour structurer le champ GraphRAG : **G-Indexing → G-Retrieval → G-Generation**. Couvre exhaustivement les types de graphes, les stratégies de retrieval, les techniques d'intégration, et les applications par domaine (dont juridique). Ce n'est pas un papier de méthode mais un travail de cartographie qui fixe le vocabulaire commun du sous-champ.

## Contributions principales

1. **Framework 3-phases** universellement repris en 2025 (Knowledge Organization, Knowledge Retrieval, Knowledge Integration).
2. **Taxonomie des KGs** en 3 types (open-domain, domain-specific, hybrid).
3. **Comparaison structurée RAG vs GraphRAG** sur 4 axes (représentation, multi-hop, scalabilité, interprétabilité).
4. **Panorama nominatif** de ~50 méthodes (Microsoft GraphRAG, ToG, LightRAG, KAG, RAPTOR, StructRAG, HYBGRAG, KnowGPT...).
5. **Identification du vide juridique** : LYNX seul KG juridique cité face à des dizaines de KGs biomédicaux.

---

## 1. Pourquoi GraphRAG ? Comparaison avec le RAG classique

### 1.1 Schéma comparatif (Figure 4 du papier)

> [!info] Figure 4 — Traditional RAG vs GraphRAG
> ![[Pasted image 20260413131636.png]]

La Figure 4 oppose en deux colonnes un pipeline **Traditional RAG** (gauche) et un pipeline **GraphRAG** (droite), sur quatre axes : Related Entities, Multi-Hop Relationships, Multi-Hop Reasoning, Answer.

### 1.2 Limites du RAG classique (§III.B)

> "Domain-specific queries often involve jargon that requires contextual comprehension. [...] Traditional RAG uses chunking to divide documents into smaller pieces to manage this complexity and improve indexing efficiency, it sacrifices critical contextual information, significantly reducing retrieval accuracy and contextual understanding."

Trois points de friction structurels :

- **Chunking aveugle** : le découpage en passages de taille fixe casse la cohérence sémantique. *"Linked/useful information may be missing."*
- **Retrieval flat sur similarité vectorielle** : top-k sur cosinus n'exprime pas les relations entre passages → multi-hop quasi impossible. *"Difficult to integrate with multiple data nodes."*
- **Contraintes inhérentes des LLMs** : fenêtre de contexte 2K-32K → chunking obligatoire mais destructeur. Sur grandes bases d'entités, *"computationally very expensive."*

### 1.3 Avantages du GraphRAG (§III.D)

> "GraphRAG provides several key advantages over traditional RAG systems, enhancing the capabilities of AI-driven information retrieval and generation."

Quatre avantages clés (sous-sections III.D.1 à III.D.4) :

1. **Enhanced Knowledge Representation** — le graphe représente hiérarchies, associations, relations multi-hop, *"revealing non-obvious connections between different pieces of information."*
2. **Flexibility in Knowledge Sources** — intègre structuré (DB), semi-structuré (JSON/XML), non-structuré (texte).
3. **Efficiency and Scalability** — graph DBs optimisées pour requêtes relationnelles. *"GraphRAG systems can generate LLM responses using 26% to 97% fewer tokens compared to traditional methods."*
4. **Interpretability** — chemins de raisonnement traçables, *"especially valuable in fields like healthcare, finance, or legal applications where decision-making processes need to be auditable."*

### 1.4 Tableau comparatif

| Critère | Traditional RAG | GraphRAG |
|---|---|---|
| **Représentation** | Chunks plats + embeddings | Graphe entités-relations typées, hiérarchies |
| **Granularité** | Chunks de taille fixe | Entités, relations, sous-graphes, communautés |
| **Retrieval** | Similarité vectorielle (top-k flat) | Similarity / Logical / GNN / LLM / RL / Multi-round / Hybrid |
| **Multi-hop reasoning** | Quasi impossible | Natif |
| **Préservation des relations** | Perdues au chunking | Encodées en triplets |
| **Traçabilité / explicabilité** | Faible (boîte noire top-k) | Forte (chemins inspectables) |
| **Scalabilité** | Coûteuse sur grandes bases | Optimisée (26-97% tokens en moins) |
| **Sources hétérogènes** | Texte non structuré surtout | Structuré + semi + non structuré |
| **Mise à jour** | Re-indexation coûteuse | Incrémentale (nœuds/arêtes) |
| **Coût d'indexation** | Faible (embeddings) | Plus élevé (extraction KG) |
| **Échec typique** | Multi-hop, jargon, contextes longs | Coût d'ingestion, qualité d'extraction |

> [!important] Synthèse
> Le RAG classique est essentiellement un **retrieval flat sur vecteurs** qui casse la structure relationnelle du corpus. GraphRAG rapatrie cette structure au niveau de la représentation : le graphe devient à la fois **l'index, le support du raisonnement multi-hop et la trace d'explicabilité**.

---

## 2. Les 3 types de Knowledge Graphs (Figure 2)

> [!info] Figure 2 — Pipeline et 3 types de KG
> ![[Pasted image 20260413131450.png]]


Le papier (§IV.C) distingue trois grandes familles de KG mobilisables dans GraphRAG.

### 2.1 Open-domain Knowledge Graphs

> "Open-domain KGs cover a wide range of domains [...] or general-purpose, like DBpedia and YAGO."

- **Schéma** : large, transversal, non spécialisé.
- **Entités** : génériques (personnes, lieux, organisations, concepts).
- **Sources** : Wikipedia, Wiktionary, Web crawls, bases ouvertes.
- **Construction** : algorithmes domain-independent (NER/RE généralistes, OpenIE).
- **Exemples** : **DBpedia**, **YAGO / YAGO 4.5**, **Wikidata**, **Freebase**, **ConceptNet**.
- **Cas d'usage** : QA ouvert, désambiguïsation, entity linking, fact-checking.
- **Limites** : bruit/redondance à grande échelle, schéma trop large pour raisonnement expert. *"The central challenge is to develop pruning algorithms that dynamically retrieve reasoning paths or subgraphs from these KGs."*

### 2.2 Domain-specific Knowledge Graphs

> "Domain-specific knowledge graphs address the need for integrating specialized knowledge from specific domains."

- **Schéma** : ontologie métier riche, contraint, finement typé.
- **Entités** : spécialisées (gènes, maladies, articles de loi, jurisprudence).
- **Sources** : bases professionnelles, articles scientifiques, manuels, dossiers patients.
- **Construction** : supervisée par experts ou semi-auto avec ontologies de référence (UMLS, MeSH, ontologies juridiques).
- **Exemples** : **UMLS**, **SPOKE**, **STRING**, **DrugBank**, **PrimeKG** (médical) ; **LYNX** (juridique multilingue) ; **AceKG** (académique).
- **Cas d'usage** : medical QA, aide au diagnostic, legal QA, compliance, recherche scientifique.
- **Limites** : construction coûteuse, dépendance experte, couverture limitée, mise à jour difficile dans des domaines évolutifs (droit, médecine).

### 2.3 Hybrid / Corpus-based Knowledge Graphs

> "Hybrid GraphRAG maintains the original textual form while utilizing graph structures primarily as an indexing mechanism to organize and retrieve relevant text chunks efficiently."

- **Schéma** : émergent, extrait par LLMs ou pipelines non supervisés.
- **Nœuds hétérogènes** : entités, concepts, mentions, **chunks de texte**, communautés.
- **Arêtes** : relations sémantiques + liens d'indexation (entity-chunk, chunk-chunk).
- **Deux sous-familles** :
  - **Knowledge-based GraphRAG** : KG explicite extrait du corpus (triplets entités-relations).
  - **Hybrid GraphRAG** : graphe comme index au-dessus des chunks, préserve le texte original.
- **Exemples** : **Microsoft GraphRAG**, **LightRAG**, **HippoRAG**, **KAG**, **StructRAG**, **HYBGRAG**, **ToG**.
- **Cas d'usage** : QA sur base documentaire privée, recherche juridique sur jurisprudence propre, assistants spécialisés à corpus évolutif.
- **Limites** : qualité dépendante de l'extraction LLM (hallucinations), coût d'indexation élevé (appels LLM), évaluation difficile (pas de gold standard).

### 2.4 Synthèse comparative

| Critère | Open-domain | Domain-specific | Hybrid / Corpus-based |
|---|---|---|---|
| **Source** | Web ouvert, Wikipedia | Ontologies métier + experts | Corpus utilisateur (texte brut) |
| **Schéma** | Large, générique | Strict, ontologie métier | Émergent, extrait par LLM |
| **Entités** | Génériques | Spécialisées | Mixtes (entités + chunks) |
| **Construction** | Pipelines généralistes | Semi-auto + curation | LLM-driven extraction |
| **Exemples** | DBpedia, YAGO, Wikidata | UMLS, SPOKE, LYNX, AceKG | MS GraphRAG, LightRAG, HippoRAG |
| **Force** | Couverture large | Précision, expertise | Adaptabilité au corpus privé |
| **Faiblesse** | Bruit, peu de profondeur | Coût, couverture limitée | Qualité variable, coût LLM |

> [!important] Pour le projet KG juridique FR
> Le projet relève à la fois du **Domain-specific** (ontologie juridique, jurisprudence Cassation) et du **Hybrid / Corpus-based** (extraction depuis arrêts bruts). Combinaison des deux explicitement recommandée par le papier.

---

## 3. Taxonomie des méthodes GraphRAG en 3 phases (Figure 3)

> [!info] Figure 3 — Taxonomie complète
> ![[Pasted image 20260413131558.png]]

Le pipeline GraphRAG est structuré en **3 phases successives**, chacune avec ses sous-catégories et méthodes représentatives.

```
GraphRAG for Customized LLMs
│
├── 1. Knowledge Organization  (= G-Indexing)
│    ├── Graphs for Knowledge Indexing  (GNN-based / LLM-based / Rule-based)
│    ├── KG Construction from Corpora   (StructRAG, GraphReader, SAIL, …)
│    ├── GraphRAG with Existing KG      (KnowGPT, ToG, RoG, GraphCoT, …)
│    └── Hybrid GraphRAG                (Graft, Tel, MoKGraphRAG, KG2RAG, HYBGRAG)
│
├── 2. Knowledge Retrieval  (= G-Retrieval)
│    ├── Retrieval Technique
│    │     ├── Similarity-based  (BERT, TF-IDF, embeddings)
│    │     ├── Logical-based     (RoG, ReP-RAG, RARAG)
│    │     ├── GNN-based         (GNN-RAG, SURGE, GNN-Ret)
│    │     ├── LLM-based         (ToG, LightRAG, KGP, TIARA)
│    │     └── RL-based          (KnowGPT, Spider, GraphRAG-RL)
│    └── Retrieval Strategy
│          ├── Once Retrieval
│          ├── Iterative Retrieval     (KGR, Tel, CoK, ReP-RAG)
│          ├── Multi-round Retrieval   (ToG, ToG 2.0, HYBGRAG)
│          ├── Post-retrieval          (StructRAG, ToG 2.0)
│          └── Hybrid Retrieval
│
└── 3. Knowledge Integration  (= G-Generation)
     ├── Fine-tuning
     │     ├── Node-level   (SKETCH, GraphGPT)
     │     ├── Path-level   (RoG, GLRec, KGTransformer, MuseGraph)
     │     └── Subgraph-level  (GRAG, GNP, InstructKGLM, LLAGA)
     └── In-context Learning
           ├── Graph-enhanced CoT      (ToG, Graph-CoT, LARK, CoK, MindMap)
           └── Collaborative KG Refinement  (KELP, FMEA-KG, EtD, CogMG)
```

### 3.1 Phase 1 — Knowledge Organization (= G-Indexing)

> "Knowledge organization refers to the systematic structuring, indexing, and integration of knowledge to facilitate efficient retrieval and reasoning by LLMs." (§IV)

- **Objectif** : pré-structurer le corpus en un support graphe exploitable.
- **Input** : corpus brut (documents, KG existants, bases hétérogènes).
- **Output** : graphe indexé + embeddings, prêt à être interrogé.

**Trois paradigmes** identifiés :

| Sous-catégorie | Objet | Méthodes |
|---|---|---|
| Graphs for Knowledge Indexing | Construire un graphe pour indexer un corpus textuel | GNN-RAG, PG-RAG, GraphCoder, KGP, MS GraphRAG, LightRAG, ToG, KAG, RoG, OG-RAG, DALK |
| KG Construction from Corpora | Extraction d'un KG depuis des documents | StructRAG, FoodKG, CoK, KGP, GraphReader, QUEST, SAIL |
| GraphRAG with Existing KG | Utilisation d'un KG existant (DBpedia, SPOKE…) | KnowGPT, ToG, ToG 2.0, LightRAG, RoG, GraphCoT, KELP |
| **Hybrid GraphRAG** | Combinaison des deux approches | Graft, Tel, MoKGraphRAG, KG2RAG, HYBGRAG |

### 3.2 Phase 2 — Knowledge Retrieval (= G-Retrieval)

> "Given a query and a graph knowledge base with dense information, retrieving factual information relevant to the given query from the knowledge base is very important in developing effective and efficient GraphRAG systems." (§V)

- **Input** : query + graphe indexé.
- **Output** : sous-graphe / chemins / nœuds pertinents à injecter au LLM.

→ **Détaillée en section 5** (techniques + stratégies + enhancement).

### 3.3 Phase 3 — Knowledge Integration (= G-Generation)

> "The integration phase focuses on seamlessly synthesizing documents obtained from knowledge retrieval into a cohesive prompt, simultaneously with appropriate training goals for the purpose of optimization." (§VI)

- **Input** : knowledge retrieved + query + LLM.
- **Output** : réponse générée alignée avec la structure graphe.

→ **Détaillée en section 6** (fine-tuning + ICL + enhancement).

---

## 4. Focus : Hybrid Graphs (notre cible de construction)

### 4.1 Définition

> "Hybrid GraphRAG: which uses graphs as knowledge carriers to knowledge indexing and integrated into a unified knowledge base, serving as a carrier of essential knowledge condensed from the raw corpora." (§IV)

> "This paradigm utilizes graph structures both as carriers of knowledge and as indexing tools. A common approach involves constructing a graph that encapsulates key information from the original text, with each node linked to corresponding text chunks. These text chunks act as a complementary knowledge source, providing detailed contextual information." (§IV.C)

> [!important] Idée-clé
> Un **hybrid graph** n'est ni un pur KG d'entités-relations, ni un simple document/chunk graph : c'est un graphe **à deux étages** où entités (et/ou communautés, résumés) sont **reliées aux chunks textuels source**. Le graphe structure la navigation ; les chunks fournissent le contexte détaillé au LLM.

### 4.2 Types de nœuds et d'arêtes combinés

| Niveau | Nœuds | Arêtes |
|---|---|---|
| **Structurel / sémantique** | Entités, concepts (extraction LLM) | Relations typées entre entités |
| **Communautaire / résumé** | Communautés (Leiden), community reports | Appartenance entité → communauté, hiérarchie de résumés |
| **Textuel / surface** | Chunks de texte, passages, documents | Lien chunk ↔ entité, liens inter-chunks (similarité, co-occurrence) |

Microsoft GraphRAG illustre cette architecture (métaphore du papier) :

> "Microsoft GraphRAG constructs a knowledge graph from private datasets, leveraging graph learning to create hierarchical summaries, detailed neighborhood reports for each of these communities, providing a multi-tiered guidebook to the entire city. This indexing phase creates a comprehensive, hierarchical understanding of the urban landscape, from individual buildings to entire districts."

### 4.3 Avantages vs alternatives pures

| Critère | KG pur | Document graph pur | **Hybrid graph** |
|---|---|---|---|
| Multi-hop structuré | Oui | Faible | **Oui** |
| Détails contextuels bruts | Non | Oui | **Oui** (via chunks) |
| Navigabilité globale (community) | Limitée | Non | **Oui** |
| Couverture domaine ouvert | Dépend du KG | Bonne | **Bonne** |
| Interpretabilité / auditabilité | Haute | Faible | **Haute** |

### 4.4 Méthodes représentatives

Cités explicitement dans la branche Hybrid GraphRAG : **Graft**, **Tel**, **MoKGraphRAG**, **KG2RAG**, **HYBGRAG**.

Systèmes hybrides de facto :
- **Microsoft GraphRAG** — KG d'entités + community reports + chunks originaux.
- **LightRAG** — dual-level (entités low-level + thèmes high-level + pointeurs vers chunks).
- **KAG** — Knowledge Augmented Generation, mixe KG et docs.
- **StructRAG** — structures hybrides adaptatives au type de question.

### 4.5 Défis spécifiques de construction

1. **Cohérence inter-niveaux** : aligner chunks ↔ entités ↔ communautés sans duplication ni dérive sémantique.
2. **Coût computationnel** : community detection + summaries (Microsoft GraphRAG) très lourd.
3. **Qualité d'extraction LLM** : bruit, hallucinations sur entités/relations.
4. **Granularité d'indexation** : trop fine = sur-fragmentation ; trop grossière = perte de précision.
5. **Maintenance incrémentale** : MAJ KG structuré + re-chunking synchronisé.
6. **Arbitrage retrieval** : choisir dynamiquement KG vs chunks vs les deux.

---

## 5. Knowledge Retrieval (détails Section V + Table 1)

### 5.1 Pipeline en 3 étapes

> [!info] Pipeline général du Knowledge Retriever
> **Query/Graph Preprocessing → Matching → Pruning → Output**

#### 5.1.1 Query / Graph Preprocessing

> "Operates simultaneously on both the query and graph databases to prepare them for efficient retrieval."

- **Côté query** : vectorisation dense ou extraction de **key-terms** (entités, relations, triplets).
- **Côté graphe** : prétraitement des composantes (entités, relations, triplets) en représentations compatibles avec la query (PLMs, graph embeddings).
- **Objectif** : produire un **espace de représentation partagé** query ↔ graphe.

#### 5.1.2 Matching

> "Matching aligns preprocessed query representations with the indexed graph database."

Quatre sous-mécanismes :
- **Semantic similarity** (embeddings, cosine).
- **Structural relationships** (alignement topologique).
- **Explicit graph-based retrieval** avec **multi-hop reasoning ability** (parcours de chemins).
- **Advanced computational components** (GNNs, LLMs encodant sémantique + structure).

→ Produit un pool de **candidate nodes / edges / subgraphs / paths**, souvent bruité.

#### 5.1.3 Knowledge Pruning

> "The pruning stage refines the initially retrieved knowledge to improve its quality and relevance."

- Filtrage des composants non pertinents.
- Consolidation et summarization des sous-graphes.
- Adaptation à la fenêtre de contexte du LLM.

> [!tip] Articulation
> Preprocessing **définit l'espace** → Matching **peuple un pool** → Pruning **produit le contexte final**. Les trois se compensent : matching grossier sauvable par pruning agressif, mais au prix de la latence.

### 5.2 Table 1 — Techniques de retrieval (extrait)

![[Pasted image 20260413131735.png]]

> [!info] Lecture transversale
> - **Granularité graphe** : entité isolée → triplets → chemins → sous-graphes complets.
> - **Output** : subgraph, reasoning path, triples, ou local context (passages rattachés).
> - **Pruning** = phase la plus diversifiée : PCST, BFS/DFS, LLM-agent scoring, RL reward, reranking.

### 5.3 Retrieval Strategies (mécanisme principal)

| Stratégie | Principe | Force | Faiblesse | Use-case typique |
|---|---|---|---|---|
| **Semantic Similarity-based** | Cosine sur embeddings ou BM25/TF-IDF | Simple, scalable | Rate les relations implicites | FAQ, retrieval dense |
| **Logical Reasoning-based** | Règles symboliques, ILP, contraintes (SPARQL) | Explicabilité, multi-hop exact | Règles à curater | KGQA formel, **droit** |
| **GNN-based** | Message-passing pour contextualiser | Dépendances structurelles | Training supervisé requis | Big KGs supervisés |
| **LLM-based** | LLM guide la sélection/exploration via prompting | Flexibilité, zero-shot, cross-domain | Coût, hallucinations | Open-domain QA |
| **RL-based** | Agent apprend à traverser ; reward = qualité du contexte | Optimise directement la qualité | Reward design délicat, exploration coûteuse | Exploration longue |
| **Hybrid** | Combine plusieurs ci-dessus | Couverture large | Complexité pipeline | Systèmes production |

**Reward typique RL-based** (cité par le papier) :
> "(i) Encompasses as many source and target entities as possible; (ii) The entities and relations within G_sub exhibit a strong relevance to question context; (iii) G_sub is concise with little redundant information such that it can be fed into LLMs with limited lengths."

### 5.4 Retrieval Enhancement Strategies (Section V-C)

| Stratégie | Mécanisme | Méthodes | Gain principal |
|---|---|---|---|
| **Multi-round Retrieval** | Itération query/refinement sur plusieurs tours | ToG, GoR, CoK, GenGraphRAG | Multi-hop, complexité |
| **Post-retrieval / Re-ranking** | Scoring post-hoc du pool | StructGPT, ToG 2.0 | Précision du contexte final |
| **Query Expansion / Rewriting** | Augmentation de la query par entités retrouvées | ToG, CoK | Rappel |
| **Self-reflection** | LLM juge la suffisance, relance si besoin (Self-RAG sur graphe) | ToG | Robustesse |
| **Adaptive Retrieval** | Profondeur modulable selon complexité estimée | RL-based, ToG | Efficacité |
| **Hybrid (multi-source)** | Fusion KG + texte + tables | ToG-2, StructRAG, HYBGRAG | Couverture |

> [!info] Trade-offs clés (auteurs)
> - **Effectiveness vs efficiency** : multi-round + LLM-based = qualité ↑ mais latence ↑.
> - **Precision vs recall** : discrete favorise précision, embedding favorise rappel.
> - **Generalization vs specialization** : GNN-based excellent en domaine entraîné, mauvais transfert ; LLM-based robuste cross-domaine au prix du coût.

---

## 6. Knowledge Integration (détails Section VI)

> "Integrating graph-retrieved knowledge into LLMs mainly includes two main ways: **fine-tuning** and **in-context learning**."

### 6.1 Fine-tuning Techniques (Table II)

Trois granularités selon la nature de l'input :

#### 6.1.1 Node-level Knowledge

- **Principe** : fine-tuning sur attributs/embeddings d'entités individuelles.
- **Use** : tâches entity-centric, KG dense.
- **Limites** : perd la structure relationnelle ; ré-entraînement à chaque évolution.
- **Méthodes** : **SKETCH** (LLaMA-2), **GraphGPT** (Baichuan-7B).

#### 6.1.2 Path-level Knowledge

- **Principe** : input = séquence de triplets sérialisée (path).
- **Use** : raisonnement multi-hop explicite, traçabilité de la chaîne d'inférence (legal, médical).
- **Limites** : coût de retrieval amont, risque d'hallucination si maillon erroné, longueur de contexte.
- **Méthodes** : **GLRec**, **KGTransformer**, **MuseGraph**, **RoG**.

> "Linguistic tasks often involve intricate reasoning and require a clear understanding of factual relationships. Utilizing knowledge graph paths, LLMs are guided through the transitory relationships and entities, thereby enhancing their reasoning capabilities with evidence-based support."

#### 6.1.3 Subgraph-level Knowledge

- **Principe** : input = sous-graphe entier, linéarisé ou encodé.
- **Use** : génération/synthèse globales, KG riches en structure locale (clusters, communautés).
- **Limites** : *"a multitude of connections beyond those found in paths [...] posing a greater challenge for LLMs to learn"* ; linéarisation lossy ; coût d'entraînement élevé.
- **Méthodes** : **RHO**, **GNP**, **InstructKGLM**, **MoleculeSTM**.

### 6.2 In-context Learning Techniques (Table III)

> "Many state-of-the-art LLMs remain closed-source in practice. The integration of closed-source LLMs is constrained since it is not feasible to jointly train or fine-tune closed-source LLMs in an end-to-end manner."

→ Voie **obligatoire** pour GPT-4, Claude, etc.

#### 6.2.1 Integration Templates (Direct prompt-based)

- Sérialisation du sous-graphe dans le prompt via template.
- *"LLMs are highly sensitive to the prompt format"* — qualité dépend du template.
- Limites : perte topologique, fenêtre de contexte, coût par requête.

#### 6.2.2 Graph-enhanced Chain-of-Thought

- **Principe** : CoT avec **navigation explicite du graphe**, hop par hop, itératif.
- **Méthodes** : **Think-on-Graph (ToG)**, **Chain-of-Knowledge (CoK)**, **MindMap**, **LARK**, **Graph-CoT**, **GNN-RAG**, **RoG**.
- **Use** : QA multi-hop complexe, explicabilité, vérification factuelle.
- **Limites** : latence (plusieurs appels LLM/question), sensible à qualité du KG.

**Exemple ToG** : à chaque étape, LLM sélectionne entités/relations prometteuses → interroge KG → étend chemin → décide continuer ou répondre (beam search piloté par LLM).

#### 6.2.3 Collaborative Knowledge Graph Refinement

- **Principe** : LLM et KG **co-raffinent** ; le KG corrige la sortie LLM, le LLM peut proposer des MAJ au KG.
- **Méthodes** : **KELP**, **KG-Rank**, **CogMG**, **EtD**, **FMEA-KG**.
- **Use** : domaines à forte exigence factuelle (legal, médical), KG évolutifs.
- **Limites** : boucle de raffinement = latence + coût ; nécessite KG de qualité.

> "Refining the LLM's original response based on the factual knowledge in knowledge graphs is also an effective method to prevent LLMs from hallucination scenarios. The timeliness and accuracy of knowledge graphs are crucial."

### 6.3 Tableau récapitulatif Fine-tuning vs ICL

| Critère | FT Node | FT Path | FT Subgraph | ICL Template | Graph-CoT | Collab. Refinement |
|---|---|---|---|---|---|---|
| **Accès aux poids** | Oui | Oui | Oui | Non | Non | Non |
| **Latence inférence** | Basse | Basse | Basse | Basse | **Haute** | Haute |
| **Coût entraînement** | Moyen | Moyen | Élevé | Nul | Nul | Nul |
| **Explicabilité** | Faible | Bonne | Moyenne | Faible | **Excellente** | Excellente |
| **Multi-hop** | Faible | Bonne | Bonne | Faible | **Excellente** | Bonne |
| **KG évolutif** | Mauvais | Mauvais | Mauvais | Bon | **Excellent** | **Excellent** |
| **Anti-hallucination** | Moyen | Bon | Bon | Moyen | Bon | **Excellent** |
| **LLM closed-source** | Non | Non | Non | Oui | Oui | Oui |

### 6.4 Integration Enhancement Strategies (§VI-C)

- **Training avec modèles domain-specific** : encodeurs spécialisés (image, molécule, code) couplés au LLM (ex. MolLM, SAIL).
- **Multi-round Integration** : raffinement itératif du contexte entre tours (IM-RAG, MedPaLM).

> [!warning] Pièges récurrents
> 1. **Sensibilité au format de prompt** : ordre des triplets, séparateur, style narratif → résultats différents.
> 2. **Fenêtre de contexte** : sérialisation naïve de gros sous-graphes → explosion 32k tokens.
> 3. **Perte topologique à la linéarisation** : préférer GNN-fusion (GraphGPT, GNP) si structure cruciale.
> 4. **Latence cumulée multi-round** : chaque hop = 1 appel LLM.
> 5. **Fraîcheur du KG** : fine-tuning fige la connaissance ; pour domaine évolutif, préférer ICL.
> 6. **Tokens rares** : *"may be treated as out-of-vocabulary tokens"* — attention au vocabulaire spécialisé.

---

## 7. Liste consolidée des graphes existants

| Type | Structure | Exemples cités | Domaine |
|---|---|---|---|
| **KG général** | Triplets typés | DBpedia, Freebase, Wikidata, YAGO, ConceptNet | QA général, web sémantique |
| **KG domain-specific** | KG typé spécialisé | SPOKE, UMLS, PrimeKG, SNOMED-CT, ICD, MedDRA | Médical, biomédical |
| | | **LYNX**, EU-law KG | **Juridique** |
| **Document graph** | Nœuds = docs/chunks ; arêtes = similarité, citation | GraphReader, KGP, QUEST | Multi-doc QA |
| **Code graph** | AST, call graph, data-flow | CodexCGraph, GraphCoder | Génération code, debug |
| **Scene / Multimodal** | Objets visuels, relations spatiales | Scene-graphs, MultiModQA | VQA, robotique |
| **Bio-scientifique** | Gènes, drogues, maladies | SPOKE, PrimeKG, DrKG, BioRED, Hetionet | Drug discovery |
| **Temporal / dynamic** | Arêtes horodatées | TempoQR, TKG | Questions temporelles |
| **Tabular-derived** | Tables → graphes relationnels | TabLLaMA, TabGraph | QA sur tables |
| **Rule / logic graph** | Règles logiques | RoG, OG-RAG | Raisonnement symbolique |
| **Community / hierarchical** | Communautés Leiden + résumés multi-niveaux | **MS GraphRAG**, Fast GraphRAG, Azure GraphRAG | QA "global queries" |
| **Hybrid** | KG + chunks + communautés liés | **MS GraphRAG**, **LightRAG**, **KAG**, **StructRAG**, **HYBGRAG**, **KG2RAG** | Entreprise, juridique, médical |

> [!important] Vide juridique
> Sur ~50 KGs cités, **un seul est juridique : LYNX** (projet H2020 EU). Pas d'équivalent SPOKE en droit. C'est à la fois une **opportunité** (champ ouvert) et un **signal d'alerte** (peut-être une difficulté intrinsèque du domaine — formalisation du raisonnement juridique, ontologie évolutive, sources hétérogènes).

---

## 8. Open challenges (§VII)

1. **Graph quality & maintenance** : garder le KG à jour.
2. **Scalability** : graphs massifs.
3. **Knowledge integration** : combiner plusieurs sources.
4. **Reasoning depth** : multi-hop sans explosion combinatoire.
5. **Domain adaptation** : customisation cross-domaine.
6. **Evaluation metrics** : benchmarks comprehensifs pour graph-augmented generation.
7. **Explainability** : traçage des chemins de raisonnement.
8. **Efficiency** : trade-off retrieval comprehensif vs vitesse.

---

## 9. Synthèse pour notre projet KG juridique français

> [!important] Recommandations directes issues du papier

### 9.1 Type de graphe à construire : **Hybrid Graph**

Cible architecturale claire : un **graphe à 3 niveaux** inspiré de Microsoft GraphRAG + LightRAG, adapté au droit français :

- **Niveau structurel** : entités juridiques typées (articles de code, décisions, parties, juridictions, notions, fondements légaux) + relations typées (fonde, casse, confirme, vise, applique).
- **Niveau communautaire** : clusters jurisprudentiels (par matière, par chambre, par motif), avec community reports générés par LLM.
- **Niveau textuel** : chunks (considérants, attendus, motivations) liés aux entités qu'ils mentionnent — **indispensable pour la traçabilité juridique** (citer le verbatim).

Ce design combine *Domain-specific KG* (ontologie juridique) et *Hybrid GraphRAG* (extraction depuis le corpus).

### 9.2 Pipeline 3-phases pour le projet

| Phase | Recommandation pour le droit FR | Justification |
|---|---|---|
| **G-Indexing** | Hybride : ontologie noyau curée (~15 classes, ~25 relations) + extraction LLM-based itérative sur arrêts Cassation | Schéma léger > schemaless (consensus papier) ; LLM-based pour la flexibilité du langage juridique |
| **G-Retrieval** | **Hybrid retriever** = Logical-based (citation d'articles, recherche par fondement) + Semantic similarity (cas analogues) + Multi-round (raffinement) | Le juriste mélange recherche par mots-clés et par analogie ; le multi-round est natif dans la pratique juridique |
| **G-Generation** | **Path-level fine-tuning** (si LLM open-source) OU **Graph-enhanced CoT** (ToG / CoK style) si GPT-4 ; toujours coupler à **Collaborative KG Refinement** | Path-level = traçabilité de la chaîne d'inférence (essentiel pour l'auditabilité légale) ; Refinement = anti-hallucination critique en droit |

### 9.3 Méthodes prioritaires à prototyper

1. **Microsoft GraphRAG** — référence d'architecture hiérarchique ; tester si les communautés Leiden ont du sens sur de la jurisprudence (clusters par matière ? par chambre ?).
2. **LightRAG** — dual-level retrieval ; aligne bien avec la dualité juridique articles vs cas.
3. **Think-on-Graph (ToG)** — beam search piloté par LLM ; testé sur le mode de raisonnement juridique multi-hop (fait → fondement → précédent).
4. **HYBGRAG / StructRAG** — verifier le pattern hybride graphe + texte avec verification post-retrieval.

### 9.4 Stratégies d'évaluation à mettre en place

Le papier identifie l'**absence de benchmarks juridiques** comme un open challenge. Conséquence pratique pour le mémoire :
- Construire un benchmark propre : ~50-100 questions juridiques de complexité variable (1-hop : "quelle peine prévue par X ?" → multi-hop : "quels arrêts récents ont infléchi la jurisprudence sur Y ?").
- Métriques : précision factuelle (citation correcte d'article/arrêt), traçabilité (chemin d'inférence présent), couverture (rappel des cas pertinents), explicabilité subjective (évaluation par juriste).

### 9.5 Risques et limites identifiés pour le projet

> [!warning] Points de vigilance
> 1. **Coût d'extraction LLM** sur l'ensemble de la jurisprudence Cassation = budget à anticiper.
> 2. **Hallucinations sur les noms d'arrêts / numéros d'articles** : risque majeur en droit ; le Collaborative KG Refinement est non-négociable.
> 3. **Évolution du droit** : le KG doit être incrémental (nouveaux arrêts), exclut le pure fine-tuning.
> 4. **Absence de gold standard juridique** : devra être construit, biais possible.
> 5. **LYNX est un projet européen abouti mais peu ouvert** : étudier ce qui peut en être réutilisé vs reconstruit.

### 9.6 Positionnement de la contribution potentielle du mémoire

Sur les 8 open challenges du papier, **3 sont particulièrement pertinents** pour positionner une contribution originale :
- **Domain adaptation** : appliquer le framework GraphRAG au droit français = peu de précédents.
- **Evaluation metrics** : benchmark juridique = vide identifié.
- **Explainability** : traçabilité de la chaîne juridique = besoin métier fort + champ encore peu travaillé.

---

## 10. Connexions

### Articles liés
- [[Peng-2024-GraphRAG-Survey-ACM]] — première survey systématique, à comparer point par point avec celle-ci.
- [[Han-2025-RAG-With-Graphs-Survey]] — autre survey holistique 2025.
- [[Microsoft-2024-GraphRAG]] — référence majeure analysée dans le survey.
- [[Awesome-GraphRAG-DEEP-PolyU]] — repo maintenu par les mêmes auteurs.
- [[Belikov-Raoult-2025-KG-Cassation]] — application juridique française directe.
- [[DAmato-2025-KG-Violence-Women-CEDH]] — autre application légale (CEDH).

### Concepts liés
- [[G-Indexing]]
- [[G-Retrieval]]
- [[G-Generation]]
- [[Hybrid Graph]]
- [[Multi-hop reasoning]]
- [[Think-on-Graph]]
- [[Community detection - Leiden]]

### Questions ouvertes pour le projet
- Quelle granularité d'entités juridiques (article entier vs alinéa vs notion) ?
- Stratégie de retrieval hybride : quels poids entre logical et semantic ?
- Faut-il un KG hiérarchique à la MS GraphRAG ou un design plus plat ?
- Comment évaluer la qualité du reasoning juridique sur graphe (panel d'experts ? benchmark synthétique ?) ?
- Réutiliser LYNX ou repartir de zéro ?

## Citations clés

> "GraphRAG provides several key advantages over traditional RAG systems, enhancing the capabilities of AI-driven information retrieval and generation."

> "GraphRAG systems can generate LLM responses using 26% to 97% fewer tokens compared to traditional methods."

> "This transparency is crucial for building trust in AI systems and is especially valuable in fields like healthcare, finance, or legal applications where decision-making processes need to be auditable."

> "LLMs are highly sensitive to the prompt format. The order of examples in in-context learning can lead to different responses."

## Notes personnelles

- Survey à lire en triangulation avec Peng 2024 et Han 2025 — vérifier la convergence du framework 3-phases.
- DEEP-PolyU = équipe à suivre (publications et repo Awesome-GraphRAG actifs).
- Les 8 open challenges = excellente grille de positionnement pour le mémoire.
- Le vide juridique (1 seul KG cité = LYNX) est l'argument fort de la pertinence du projet.
- L'architecture hybride à 3 niveaux (entités + communautés + chunks) est la cible — elle réconcilie la précision factuelle (chunks → verbatim) et le raisonnement structuré (KG → multi-hop).
