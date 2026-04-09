---
tags: [projet, roadmap]
created: 2026-04-09
modified: 2026-04-09
---

# Roadmap — Knowledge Graph Juridique Francais

## Phase 0 : Exploration & Etat de l'art

> Comprendre le paysage, identifier ce qui existe, pourquoi, et ce qu'on peut apporter.

- [x] Identifier les axes de recherche
- [x] Recherche bibliographique initiale (50 sources)
- [x] Construire le tableau etat de l'art
- [ ] Lire les papiers — KG dans le domaine juridique (9 articles)
- [ ] Lire les papiers — Ontologies & Standards (7 refs)
- [ ] Lire les papiers — GraphRAG (8 articles)
- [ ] Lire les papiers — Construction de KG avec LLMs (3 articles)
- [ ] Lire les papiers — NLP juridique & NER (7 articles)
- [ ] Lire les papiers — Agents LLM juridiques (4 articles)
- [ ] Lire les papiers — Prediction & Citations (5 articles)
- [ ] Lire les papiers — Benchmarks & Evaluation (2 refs)
- [ ] Lire les papiers — Donnees ouvertes FR (5 refs)
- [ ] Rediger les fiches de lecture pour chaque article
- [ ] Synthese par thematique (dans 06-Analyses)
- [ ] Identifier les gaps dans la litterature
- [ ] Identifier les contributions possibles de notre projet

## Phase 1 : Choix de l'ontologie & Preparation des donnees

> Definir ce qu'on veut representer et preparer la matiere premiere.

- [ ] Etudier les ontologies existantes (ELI, ELI-I, LKIF, ECLI, AKN, ALLOT)
- [ ] Identifier le perimetre du graph : JP seule, articles + JP, ou plus
- [ ] Choisir / adapter l'ontologie pour le KG
- [ ] Definir les entites (articles de loi, decisions, moyens, juridictions...)
- [ ] Definir les relations (cite, applique, abroge, confirme, casse...)
- [ ] Explorer l'API Judilibre — comprendre la structure des donnees
- [ ] Explorer l'API Legifrance — comprendre la structure des donnees
- [ ] Parser des articles de loi et decisions (echantillon)
- [ ] Evaluer la qualite de la donnee brute
- [ ] Preparer le pipeline de nettoyage / structuration
- [ ] Formaliser l'ontologie (OWL / SHACL / schema Neo4j)
- [ ] Valider avec des exemples reels

## Phase 2 : Construction de graphs (prototypage)

> Tester la construction de graphs a petite echelle, iterer.

- [ ] Definir le pipeline d'extraction (NER + relation extraction)
- [ ] Choisir les outils : JuriBERT, LLM local, ou combinaison
- [ ] Tester la construction de triplets sur un petit corpus
- [ ] Charger dans Neo4j et visualiser
- [ ] Evaluer la qualite du graph vs attendu
- [ ] Iterer sur les prompts / la pipeline d'extraction
- [ ] Construire des graphs plus larges progressivement
- [ ] Documenter les resultats et les choix

## Phase 3 : Benchmark des LLMs sur le droit francais

> Evaluer les modeles disponibles sur des taches juridiques francaises.

- [ ] Recuperer des papiers sur les methodes d'evaluation de LLM en droit francais
- [ ] Inventorier les benchmarks existants (ou les adapter)
- [ ] Si necessaire, construire / ameliorer un benchmark
- [ ] Deployer des LLMs en local sur les clusters GPU
- [ ] Tester et benchmarker les modeles (JuriBERT, CamemBERT legal, LLMs generaux)
- [ ] Comparer les resultats
- [ ] Analyser les forces / faiblesses par type de tache juridique
- [ ] Rediger les resultats

## Phase 4 : Relations & Data Science

> Enrichir le graph avec des methodes de data science.

- [ ] Creer des relations entre articles cites en commun
- [ ] Explorer les methodes de link prediction (GNN, node2vec...)
- [ ] Donner du poids aux noeuds (centralite, PageRank juridique)
- [ ] Detecter des communautes de decisions / articles
- [ ] Evaluer la qualite des relations predites
- [ ] Iterer sur les methodes

## Phase 5 : Exploitation du graph — Recherche juridique

> Utiliser le graph pour repondre a des questions juridiques.

- [ ] Concevoir les patterns de traversee du graph
- [ ] Implementer le GraphRAG / multi-hop retrieval
- [ ] Identifier des benchmarks de comparaison (RAG vs GraphRAG vs multi-hop)
- [ ] Tester sur des questions juridiques reelles
- [ ] Evaluer et comparer les approches
- [ ] Documenter les resultats

## Phase 6 : Redaction du memoire

- [ ] Rediger chaque section
- [ ] Figures et schemas d'architecture
- [ ] Generer la bibliographie
- [ ] Relecture et finalisation
