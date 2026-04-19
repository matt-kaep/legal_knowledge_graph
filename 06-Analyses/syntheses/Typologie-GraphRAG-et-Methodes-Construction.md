---
tags: [synthese, graphrag, kg, construction]
created: 2026-04-10
modified: 2026-04-10
sources: ["Zhang-2025-GraphRAG-Survey-Customized-LLMs", "DAmato-2025-KG-Violence-Women-CEDH", "Belikov-Raoult-2025-KG-Cassation"]
statut: "vivant"
---

# Typologie des GraphRAG et methodes de construction de KG

> Etat de l'art consolide a partir des papiers lus et des stubs du repo. Document vivant, a mettre a jour au fil des lectures.

---

## 1. Typologie des types de graphs pour GraphRAG

Trois grandes familles se degagent de la litterature recente, avec des criteres de distinction bien etablis.

### 1.1 Knowledge-based GraphRAG

**Principe** : Le graphe *porte* la connaissance. Les noeuds sont des entites (articles, decisions, personnes, concepts juridiques), les aretes sont des relations semantiques typees (`cite`, `applique`, `casse`, `abroge`).

**Construction**
- Extraction d'entites et de relations depuis un corpus source
- Ou reutilisation d'un KG existant (DBpedia, YAGO, SPOKE, LYNX)

**Retrieval**
- Traversal du graphe : multi-hop, path-based
- SPARQL pour les requetes precises
- Community detection pour le raisonnement global

**Forces**
- Raisonnement multi-hop puissant
- Explicabilite (on peut tracer le chemin)
- Requetes structurees precises

**Faiblesses**
- Construction couteuse (ontologie + extraction)
- Perte d'information par rapport au texte original
- Sensible a la qualite de l'extraction

**Exemples references dans le repo**
- [[Belikov-Raoult-2025-KG-Cassation]] — KG penal de la Cour de cassation, ontologie custom
- [[DAmato-2025-KG-Violence-Women-CEDH]] — KG CEDH bottom-up (583 KB d'ontologie)
- [[Li-2024-CLKG-Construction-LLM]] — Chinese Legal KG
- [[Colombo-2025-US-Legislative-Graph]] — KG legislation US
- [[Microsoft-2024-GraphRAG]] — reference industrielle

---

### 1.2 Index-based GraphRAG (aka Document Graph)

**Principe** : Le graphe sert d'*index* pour retrouver des chunks textuels bruts. Les noeuds sont des passages de texte (paragraphes, sections, decisions), les aretes representent des similarites semantiques ou des references croisees.

**Construction**
- Chunking du corpus (decoupage en passages)
- Embedding de chaque chunk
- Graph de similarite (kNN) + liens explicites (citations, footnotes)

**Retrieval**
- Vector search sur les noeuds (embeddings)
- Expansion via les aretes pour trouver les chunks lies
- Pas de raisonnement structure, juste du retrieval enrichi

**Forces**
- Construction legere, pas d'ontologie
- Preserve la fidelite au texte original
- Rapide a mettre en place

**Faiblesses**
- Raisonnement limite (surface seulement)
- Pas de requetes structurees
- Passe mal a l'echelle en multi-hop

**Exemples references dans le repo**
- [[Yang-2024-Legal-Document-RAG-MultiAgent]] — graph de clauses + definitions juridiques
- [[47Billion-2025-GraphRAG-Legal-Reasoning]] — POC Neo4j + Qdrant (mais avec triplets, donc hybride en pratique)

---

### 1.3 Hybrid GraphRAG

**Principe** : Combine les deux approches. Le KG structure fournit le raisonnement, le document graph fournit la fidelite textuelle. Les deux sont relies : un noeud KG pointe vers les chunks qui l'ont genere.

**Architecture typique**
```
  [KG]                    [Document Graph]
  Articles    <----->     Chunks de decisions
  Decisions                Sections d'articles
  Principes                Passages cites
     |
     v
  Noeuds KG lies a leurs chunks sources
```

**Retrieval**
- Query → resolution dans le KG → traversal multi-hop
- Pour chaque noeud retrouve, recuperation des chunks sources associes
- Concatenation et injection dans le prompt LLM

**Forces**
- Meilleur des deux mondes : raisonnement structure + citations precises
- Explicabilite complete (quel article, quelle decision, quel passage)
- Adaptabilite : on peut privilegier la structure ou le texte selon la question

**Faiblesses**
- Construction la plus couteuse
- Complexite de maintenance (deux structures a synchroniser)
- Necessite une ontologie ET un bon chunking

**Exemples references dans le repo**
- [[Microsoft-2024-GraphRAG]] — extraction d'entites + community summaries + liens vers sources
- [[ArXiv-2026-RAG-KG-NMF-Legal]] — RAG + KG + Vector Store juridique US
- **MedGraphRAG** (biomedical) — reference du domaine
- **HybGRAG** — cite dans Zhang 2025

---

### 1.4 Grille de comparaison

| Critere | Knowledge-based | Index-based | Hybrid |
|---|---|---|---|
| **Noeuds** | Entites typees | Chunks de texte | Les deux |
| **Aretes** | Relations semantiques | Similarite, references | Les deux + liens croises |
| **Ontologie** | Obligatoire | Non | Obligatoire |
| **Raisonnement multi-hop** | Excellent | Limite | Excellent |
| **Fidelite au texte** | Lossy | Preserve | Preserve |
| **Cout de construction** | Eleve | Faible | Tres eleve |
| **Cout de maintenance** | Eleve | Faible | Tres eleve |
| **Explicabilite** | Structurelle | Textuelle | Complete |
| **Scalabilite** | Difficile | Facile | Moyenne |
| **Meilleure pour** | QA structure, reasoning | Retrieval factuel | QA complexe |

---

## 2. Structures organisationnelles des graphs

Orthogonal a la typologie 1.1-1.3, on peut organiser le graph de differentes manieres.

### 2.1 Flat graph

Tous les noeuds sont au meme niveau. Simple, direct. Fonctionne bien pour des corpus de quelques milliers de documents.

### 2.2 Hierarchical graph (communautes)

**Approche Microsoft GraphRAG** :
```
Niveau 3 : Resumes de communautes (vue macro)
              ^
Niveau 2 : Communautes detectees (Leiden clustering)
              ^
Niveau 1 : Entites et relations atomiques
```

**Avantages**
- Retrieval coarse-to-fine (on commence large, on descend)
- Scale sur gros corpus (480K decisions Judilibre)
- Resumes pre-generes par LLM (indexing couteux mais retrieval rapide)

**Inconvenients**
- Indexing tres couteux
- Sensible a la qualite du clustering

### 2.3 Temporal graph

Le graph evolue dans le temps. Versioning des articles de loi (abrogations, modifications), historique des jurisprudences. Crucial en droit ou un article peut etre modifie, une decision peut etre cassee.

Exemple : [[Colombo-2025-US-Legislative-Graph]] — graphe temporel montrant l'evolution legislative US.

### 2.4 Heterogeneous graph

Plusieurs types de noeuds (article, decision, juge, principe, moyen) et plusieurs types d'aretes. C'est la norme en droit. Oppose a un graphe homogene (un seul type de noeud, ex: citation network pur).

---

## 3. Methodes de construction de KG

### 3.1 Approches symboliques / rule-based

#### 3.1.1 Parsing structurel

**Principe** : Exploiter la structure formelle des documents juridiques (titres, articles, alineas, zones identifiees).

**Techniques**
- Regex sur patterns juridiques (`article L. XXX-X du Code de Y`)
- Parseurs XML pour les formats structures (ELI, AKN)
- Extraction des zones Judilibre (moyens, motivations, dispositif)

**Forces** : deterministe, rapide, tres precis sur les elements structures
**Faiblesses** : limite aux informations explicitement structurees

**Exemples dans le repo**
- [[REGLEX-Fondamentaux]] — extraction de references jurisprudentielles par parsing
- [[API-Judilibre-Cour-Cassation]] — zones pre-decoupees disponibles via l'API

#### 3.1.2 OpenIE (Open Information Extraction)

**Principe** : Extraction automatique de triplets `(sujet, predicat, objet)` sans schema prealable.

**Outils historiques**
- **TextRunner** (Yates 2007) — pionnier
- **OpenIE6** (Kolluru 2020) — moderne
- **Stanford OpenIE**, **ReVerb**, **OLLIE**

**Forces** : schemaless, rapide
**Faiblesses** : triplets bruts, pas de canonicalisation, mal adapte au juridique

#### 3.1.3 Named Entity Recognition (NER) + Relation Extraction (RE)

**Principe** : Deux etapes sequentielles : d'abord detecter les entites, puis extraire les relations entre elles.

**Modeles classiques**
- CRF, BiLSTM-CRF pour NER
- **JuriBERT** (francais juridique) — [[Douka-2021-JuriBERT-French-Legal]]
- **Legal CamemBERT** (Doctrine, Maastricht)
- **LegNER** — [[Karamitsos-2025-LegNER-Legal-NER]]

**Relation extraction**
- Classification supervisee (etant donne deux entites, predire la relation)
- Necessite des donnees annotees

**Forces** : bonne precision avec modeles domain-adapted
**Faiblesses** : pipeline sequentiel → propagation d'erreurs

---

### 3.2 Approches LLM-based

#### 3.2.1 Zero-shot prompting

**Principe** : Donner au LLM le texte + la tache (extraire les entites et relations) sans exemple.

**Exemple de prompt**
```
Extrais les entites juridiques et leurs relations du texte suivant.
Retourne un JSON avec les champs: entities, relations.

Texte: "{decision}"
```

**Forces** : simple, flexible, aucun training
**Faiblesses** : resultats variables, hallucinations, format non garanti

**Ou on l'a vu** : [[DAmato-2025-KG-Violence-Women-CEDH]] (score 61.5% sur CQ)

#### 3.2.2 Few-shot prompting

**Principe** : Ajouter 3-5 exemples (input, output) dans le prompt pour guider le modele.

**Forces** : meilleure precision que zero-shot
**Faiblesses** : consomme du contexte, sensible aux exemples choisis

**Ou on l'a vu** : [[Colombo-2025-US-Legislative-Graph]] (LLM fine-tune + few-shot)

#### 3.2.3 Schema-guided prompting

**Principe** : Donner au LLM un schema/ontologie prealable et lui demander de remplir les instances.

**Exemple de prompt**
```
Voici une ontologie juridique :
- Classes : Article, Decision, Moyen, Principe
- Relations : cite, applique, casse

Extrais du texte ci-dessous les instances conformes a cette ontologie.
```

**Forces** : sortie structuree, alignement avec l'ontologie
**Faiblesses** : demande une ontologie prealable de qualite

**Ou on l'a vu** : [[Zhang-2025-GraphRAG-Survey-Customized-LLMs]] (approche dominante actuelle)

#### 3.2.4 Fine-tuning

**Principe** : Entrainer un LLM sur un dataset d'extraction de KG specifique au domaine.

**Variantes**
- SFT (Supervised Fine-Tuning) sur triplets annotes
- Prefix-tuning (moins de parametres) — [[Li-2024-CLKG-Construction-LLM]]
- DPO pour aligner sur des preferences (ex: triplets corrects vs incorrects)

**Forces** : meilleure precision, coherence
**Faiblesses** : demande un dataset annote, cout compute

**Ou on l'a vu** : [[Guha-2026-KG-Assisted-LLM-Legal-Reasoning]] (SFT + DPO)

#### 3.2.5 Chain-of-thought / Reasoning-based extraction

**Principe** : Decomposer l'extraction en etapes de raisonnement explicites.

**Exemple**
```
1. D'abord, identifie les acteurs juridiques mentionnes.
2. Ensuite, pour chaque acteur, identifie ses actions juridiques.
3. Enfin, structure cela en triplets.
```

**Forces** : meilleure qualite sur extractions complexes
**Faiblesses** : long, couteux

#### 3.2.6 Metacognitive / Self-reflective prompting

**Principe** : Le LLM genere, puis critique, puis corrige ses propres extractions.

**Exemple dans le repo**
- [[Lippolis-2025-Ontogenia-Metacognitive]] — auto-reflexion pour generer des ontologies
- Utilise des Ontology Design Patterns (ODP) comme guides

---

### 3.3 Approches hybrides (etat de l'art)

Les meilleurs systemes combinent plusieurs methodes.

#### 3.3.1 Pipeline bottom-up + LLM

**Principe** (vu dans [[DAmato-2025-KG-Violence-Women-CEDH]])
1. Experts + analyse manuelle → ontologie noyau
2. LLM peuple les instances avec le schema en contexte
3. Validation par Competency Questions

**Forces** : precision haute + scale
**Faiblesses** : couteux

#### 3.3.2 Extract-Define-Canonicalize (EDC)

**Principe** ([[Zhang-2024-EDC-Extract-Define-Canonicalize]])
1. **Extract** : LLM extrait des triplets libres (schemaless)
2. **Define** : LLM genere des definitions des relations extraites
3. **Canonicalize** : Alignement avec une ontologie cible (deduplication, normalisation)

**Forces** : ne demande pas d'ontologie prealable, genere un schema emergent
**Faiblesses** : qualite dependante de la phase de canonicalisation

#### 3.3.3 Multi-agent extraction

**Principe** : Plusieurs agents LLM specialises collaborent (un pour NER, un pour RE, un pour validation, un pour canonicalisation).

**Exemples**
- [[Yang-2024-Legal-Document-RAG-MultiAgent]] — systeme multi-agent avec graph de clauses
- [[ArXiv-2026-LLM-Agents-Law-Taxonomy]] — taxonomie des agents juridiques

**Forces** : modulaire, ameliore les performances par specialisation
**Faiblesses** : complexite, cout

---

### 3.4 Reutilisation de KGs existants

Plutot que de construire de zero, on peut reutiliser des KGs existants :

| KG existant | Usage possible pour notre projet |
|---|---|
| **ELI / ELI-I** | Identifiants et liens entre textes legislatifs FR et EU |
| **ECLI** | Identifiants uniques pour la jurisprudence FR/EU |
| **LKIF Core** | Concepts juridiques fondamentaux |
| **LYNX** | KG juridique multilingue EU |
| **Wikidata** | Entites externes (personnes, lieux, organisations) |
| **EuroVoc** | Vocabulaire thematique EU, SKOS |

**Pattern** : utiliser ces KGs comme **socle** (entites et relations canoniques) et y **ajouter une couche domaine** specifique a notre projet.

Ref : [[ELI-European-Legislation-Identifier]], [[LKIF-Core-Legal-Knowledge-Interchange]], [[Lynx-2019-Legal-KG-Smart-Compliance]]

---

## 4. Grille de comparaison des methodes de construction

| Methode | Precision | Scale | Flexibilite | Cout | Domain fit | Hallucinations |
|---|---|---|---|---|---|---|
| Parsing structurel | Tres haute | Haute | Faible | Faible | Haute | Aucune |
| OpenIE | Moyenne | Haute | Tres haute | Faible | Faible | Faible |
| NER + RE (classique) | Haute | Moyenne | Faible | Moyen | Haute | Aucune |
| LLM zero-shot | Variable | Haute | Tres haute | Faible | Moyenne | Significatives |
| LLM few-shot | Bonne | Haute | Tres haute | Moyen | Haute | Moderees |
| LLM schema-guided | Haute | Haute | Moyenne | Moyen | Tres haute | Faibles |
| LLM fine-tune | Tres haute | Haute | Faible | Eleve | Tres haute | Faibles |
| Bottom-up + LLM | Tres haute | Moyenne | Moyenne | Eleve | Tres haute | Tres faibles |
| EDC | Bonne | Haute | Haute | Moyen | Moyenne | Moderees |
| Multi-agent | Tres haute | Moyenne | Haute | Tres eleve | Haute | Faibles |

---

## 5. Grille de decision pour notre projet

### Pour le type de graph

| Critere | Notre cas | Conclusion |
|---|---|---|
| Volume cible | 480K decisions + codes | Scale obligatoire |
| Besoin de raisonnement juridique | Fort | Structure KG necessaire |
| Besoin de citations precises | Fort | Document graph aussi |
| Besoin d'ontologie | Oui (ELI/LKIF comme socle) | Schema-based |
| **→ Type** | | **Hybrid GraphRAG** |

### Pour la structure

| Critere | Notre cas | Conclusion |
|---|---|---|
| Volume | 480K decisions | Plat difficilement gerable |
| Diversite thematique | Forte (civil, penal, commercial, travail, admin) | Communautes naturelles |
| Evolution temporelle | Oui (abrogations, revirements) | Temporal au moins partiellement |
| **→ Structure** | | **Hierarchical + temporal** |

### Pour la methode de construction

| Critere | Notre cas | Conclusion |
|---|---|---|
| Metadonnees structurees | Oui (Judilibre zones, ELI) | Parsing + API en priorite |
| Relations fines en langage naturel | Oui (moyens, motivations) | LLM necessaire |
| Qualite critique | Oui (usage juridique) | Pas de LLM seul |
| Budget annotation | Limite | Pas de fine-tuning massif |
| **→ Methode principale** | | **Hybrid : parsing structurel + LLM schema-guided few-shot** |
| **→ Methode complementaire** | | **Bottom-up pour l'ontologie, validation par CQs** |

---

## 6. Questions ouvertes

Liees a cette synthese :

- [ ] Quelle granularite des noeuds ? (decision entiere vs moyens individuels vs triplets d'arguments)
- [ ] Faut-il integrer un LLM fine-tune FR juridique (JuriBERT) ou un LLM generaliste suffit ?
- [ ] Comment gerer la temporalite dans un property graph (Neo4j) vs un triple store RDF ?
- [ ] Quelle est la taille minimale d'ontologie utile (10 classes ? 50 ? 100 ?)
- [ ] Comment evaluer la qualite du graph a grande echelle ?

---

## 7. Sources

### Papiers lus en detail
- [[Zhang-2025-GraphRAG-Survey-Customized-LLMs]] — framework 3-phases, typologie
- [[DAmato-2025-KG-Violence-Women-CEDH]] — comparaison bottom-up vs LLM
- [[Belikov-Raoult-2025-KG-Cassation]] — KG penal FR

### Papiers pertinents non encore lus (stubs)
- [[Peng-2024-GraphRAG-Survey-ACM]] — premiere survey GraphRAG
- [[Han-2025-RAG-With-Graphs-Survey]] — survey holistique 2025
- [[Microsoft-2024-GraphRAG]] — reference industrielle hierarchique
- [[Bian-2025-LLM-KG-Construction-Survey]] — survey construction KG par LLM
- [[Lippolis-2025-Ontogenia-Metacognitive]] — prompting metacognitif
- [[Zhang-2024-EDC-Extract-Define-Canonicalize]] — framework EDC
- [[Scaffidi-2025-GraphRAG-KG-Schema-Impact]] — impact du schema sur GraphRAG
