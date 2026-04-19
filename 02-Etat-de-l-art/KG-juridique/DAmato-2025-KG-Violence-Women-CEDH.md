---
tags: [article, kg-juridique]
categorie: "KG-juridique"
titre_complet: "Automated Creation of the Legal Knowledge Graph Addressing Legislation on Violence Against Women: Resource, Methodology and Lessons Learned"
auteurs: "d'Amato, Rubini, Didio, Francioso, Amara, Fanizzi"
annee: 2025
type: "Article (arXiv)"
venue: "arXiv"
url: "arxiv.org/abs/2508.06368"
doi: "10.48550/arXiv.2508.06368"
pdf_local: ""
status: "lu"
pertinence: "haute"
created: 2026-04-09
modified: 2026-04-09
---

# Automated Creation of Legal KG: Violence Against Women (CEDH)

> [!info] Metadonnees
> **Auteurs** : d'Amato, Rubini, Didio, Francioso, Amara, Fanizzi
> **Annee** : 2025 | **Venue** : arXiv
> **Type** : Article (arXiv)
> **URL** : [Lien](https://arxiv.org/abs/2508.06368)
> **Code** : [EVA-KG (bottom-up)](https://github.com/PeppeRubini/EVA-KG) | [PreJust4Womans (LLM)](https://github.com/Fra3005/PreJust4Womans) | [SPARQL endpoint](https://github.com/khaoulafatima/PJ4W)

## Resume

Construction d'un KG juridique sur les decisions de la CEDH concernant les violences faites aux femmes. Comparaison de deux approches complementaires : bottom-up systematique (scraping + parsing + RDFLib) et LLM-based (RAG + GPT-4o + Mixtral 8x22b). Le KG bottom-up produit 10 325 triples a partir de 7 373 cas (65 jugements + 8 decisions). L'approche LLM ne produit qu'une T-box (schema sans instances).

## Contributions principales

1. Comparaison systematique de deux approches de construction de KG juridique (bottom-up vs LLM)
2. Methodologie CQ-driven (13 Competency Questions) pour guider la conception de l'ontologie
3. Ressource FAIR publiee (Zenodo DOI, LOD Cloud, GitHub, SPARQL endpoint, CC BY 4.0)

## Methodologie

### Donnees
- Source : ECHR / HUDOC (hudoc.echr.coe.int)
- Volume : 7 373 cas (65 jugements + 8 decisions)
- Langue : Anglais uniquement

### Pipeline Bottom-up (6 etapes)
1. **Data collection** : Scraping Selenium 4 → PDF + HTML depuis HUDOC
2. **Knowledge extraction** : Beautiful Soup → instances ECHRDocument → RDFLib
3. **Triple integration** : Elimination des doublons → 10 325 triples
4. **Ontology creation** : 13 Competency Questions → extension ECLI + DCTERMS + Wikidata
5. **KG construction** : Integration + liens Wikidata + visu Neo4j/PyVis/RDF Grapher
6. **SPARQL endpoint** : Flask + SPARQL 1.1

### Pipeline LLM
1. **Document preparation** : Full-text (PDF complet) ou sub-part (sections par expert)
2. **RAG** : Together Embeddings (BERT-M2) + FAISS via LangChain
3. **Ontologie** : GPT-4o (generation zero-shot) + Mixtral 8x22b (enrichissement)
4. **KG** : Few-shot prompting par document puis merge
5. **Validation CQ** : Mixtral zero-shot + verification manuelle

## Resultats cles

### KG Bottom-up

| Metrique | Valeur |
|---|---|
| Triples | 10 325 |
| Entites distinctes | 5 185 |
| Predicats distincts | 2 222 |
| Taille ontologie | 583.7 KB |

### KG LLM

| Metrique | Valeur |
|---|---|
| Classes | 12 |
| Object properties | 9 |
| Data properties | 17 |
| Taille ontologie | 6.4 KB |
| Instances | **Aucune (T-box only)** |

### Scores Competency Questions (LLM, sur 5 documents)

| Question | Full-text | Sub-part |
|---|---|---|
| Cas associe a violation | 3/5 | 4/5 |
| Issue juridique du cas | 4/5 | 4/5 |
| Raison du jugement | 4/5 | 2/5 |
| Abus lie au jugement | 5/5 | 4/5 |
| Severite de l'abus | 2/5 | 2/5 |
| Duree/frequence abus | 3/5 | 2/5 |
| Articles violes | 4/5 | 4/5 |
| Contexte de l'abus | 5/5 | 4/5 |
| Juridiction | 2/5 | 1/5 |
| Dommages | 1/5 | 3/5 |
| Consequences | 5/5 | 4/5 |
| Montants | 0/5 | 4/5 |
| Statut juridique | 5/5 | 3/5 |
| **Total** | **40/65 (61.5%)** | **37/65 (56.9%)** |

### Comparaison des approches

| Critere | Bottom-up | LLM |
|---|---|---|
| Precision | Haute, alignee domaine | Variable, hallucinations |
| Temps | Laborieux | Rapide, scalable |
| Ontologie | Riche (583 KB) | Minimale (6.4 KB) |
| Use case | Raisonnement formel, SPARQL | Exploration, prototypage rapide |
| Hallucinations | Aucune | Significatives |

## Points forts

- Comparaison rigoureuse de deux paradigmes sur le meme corpus
- Methodologie CQ-driven reproductible
- Code et donnees publies (FAIR)
- Reutilisation de vocabulaires standards (DCTERMS, ECLI, Wikidata)

## Limites

- Corpus anglais uniquement — pas transposable directement au francais
- Petit corpus : 65 jugements + 8 decisions
- L'approche LLM ne peuple pas d'instances (T-box only)
- Pas de benchmark comparatif avec d'autres systemes
- Score LLM de 61.5% mediocre, surtout sur les donnees numeriques (montants : 0/5)

## Liens avec mon projet

> [!important] Pertinence pour le KG juridique francais
> Papier methodologiquement tres utile. La comparaison bottom-up vs LLM est exactement le type d'evaluation qu'on devra faire. La methodologie CQ-driven est a adopter. En revanche, le corpus (CEDH anglais) est eloigne du notre (Cour de cassation francaise).

### Ce que je peux reutiliser
- La methodologie CQ-driven pour construire l'ontologie
- Le pattern de reutilisation de vocabulaires existants (DCTERMS, ECLI, Wikidata URIs)
- La structure de comparaison bottom-up vs LLM
- Les repos GitHub comme reference d'implementation
- Le principe FAIR pour la publication du KG

### Ce que je dois adapter
- Source de donnees : passer de HUDOC/CEDH a Judilibre/Legifrance
- Langue : francais au lieu d'anglais
- Ontologie : adapter au droit francais (pas au droit CEDH)
- L'approche LLM doit aller au-dela de la T-box et peupler des instances

## Connexions

### Articles lies
- [[Belikov-Raoult-2025-KG-Cassation]] — meme approche (KG juridique) mais sur la Cour de cassation francaise
- [[LegalRuleML-OASIS]] — vocabulaire reference
- [[LKIF-Core-Legal-Knowledge-Interchange]] — ontologie de base
- [[ECLI-European-Case-Law-Identifier]] — standard reutilise dans leur ontologie
- [[Lynx-2019-Legal-KG-Smart-Compliance]] — cite comme travail connexe

### Concepts lies
- [[Competency Questions (CQ)]]
- [[T-box vs A-box]]
- [[FAIR principles]]
- [[RAG (Retrieval-Augmented Generation)]]

### Questions soulevees
- Comment combiner bottom-up et LLM dans un pipeline unique ?
- Quel score CQ viser pour notre KG ?
- Comment gerer les hallucinations LLM dans l'extraction juridique ?

## Citations cles

> "The two methodologies resulted complementary: the former providing more precise outcomes but more time-consuming, the latter more scalable but limited in accuracy"

> "The LLM struggled to independently create a complete domain-specific ontology using only its pretrained knowledge, requiring additional domain-specific documents"

> "The automatic generated ontology, while efficient and scalable, was minimal and included generic classes due to the absence of domain specific constraints"

## Notes personnelles

- Confirme Belikov & Raoult : l'ontologie doit etre construite manuellement, pas deleguee au LLM
- La methodologie CQ-driven est a adopter absolument pour notre Phase 1
- Le score de 61.5% du LLM est un point de reference — on devrait viser mieux
- L'approche hybride (bottom-up pour le schema + LLM pour le peuplement) = consensus emergent
- Les repos GitHub sont a explorer pour s'inspirer de l'implementation
- Notre projet avec Judilibre (480K+ decisions) sera d'une autre echelle que leurs 65 jugements — l'approche LLM sera quasi obligatoire pour le peuplement
