---
tags: [presentation, projet, superviseur]
type: presentation
semaine: 1
date: 2026-04-13
audience: superviseur-de-stage
format: slides-latex
status: en-cours
modified: 2026-04-13
---

# Présentation Superviseur — Week 1 (2026-04-13)

> [!info] Objectif
> Présenter l'avancement de mi-stage : travaux réalisés, contributions envisagées, roadmap.
> Format : slides LaTeX (Beamer).

> [!note] Fichier source LaTeX
> [[Week-1-2026-04-13-Presentation-Superviseur.tex|Slides Beamer (.tex)]]
>
> Compilation : `pdflatex Week-1-2026-04-13-Presentation-Superviseur.tex` (x2 pour la TOC)

---

## Plan d'ensemble

1. Titre & présentation
2. Rappel du besoin (structure de l'argument juridique)
3. Problématiques
4. État de l'art — cartographie
5. GraphRAG — concepts clés
6. Papiers lus en détail (6a, 6b, 6c, 6d)
7. Points saillants transversaux
8. Gaps identifiés
9. Positionnement & contribution
10. Roadmap vue d'ensemble
11. Prochaines étapes (court terme)
12. Annexes

---

## Slide 1 — Titre & présentation

- Sujet, encadrement, durée
- Objectif du stage en 1 phrase

---

## Slide 2a — Comment raisonne un juriste ? (exemple concret)

**Pitch en 3 phrases**
> Pour gagner un argument juridique, il faut 3 choses :
> 1. Une **règle** (la loi, un principe)
> 2. Des **faits** (ce qui s'est passé)
> 3. Une **démonstration** que la règle s'applique aux faits
>
> La **jurisprudence** (décisions passées) fait le pont entre les deux.

**Exemple — le voisin bruyant**
- *Faits* : mon voisin fait du bruit tous les soirs à 23h
- *Règle* (art. 1240 C. civ.) : "toute faute causant un dommage oblige à réparer"
- *Problème* : est-ce que "bruit à 23h" = une faute ?
- *Jurisprudence* : Cass. 2018 a jugé que des nuisances sonores répétées après 22h constituent un trouble anormal
- *Argument complet* : règle + JP + faits → conclusion

**Analogie pour un chercheur**

| Science | Droit |
|---|---|
| Théorie | Règle (article) |
| Expériences passées | Arrêts passés (JP) |
| Observation | Faits de l'espèce |
| Prédiction validée | Argument juridique |

> [!tip] Punchline
> Un juriste qui plaide = un chercheur qui doit citer **tous les articles et toutes les expériences passées pertinentes** pour convaincre.

---

## Slide 2b — Architecture du raisonnement & outil cible

**Architecture réelle** (schéma graphe)
- Argument ↔ Article (fondement)
- Argument ↔ Arrêt (JP qui applique/interprète)
- Arrêt ↔ Article (cite, applique, écarte)
- Arrêt ↔ Arrêt (confirme, casse, revirement)

> Important : la JP ne sert pas qu'à *valider* un article — elle peut aussi l'interpréter, qualifier les faits, voire créer la règle (ex : enrichissement sans cause avant 2016).

**Pourquoi c'est dur à automatiser**
- Un article seul ne suffit jamais (interprétation)
- 10 arrêts disent oui, 3 disent non → tout connaître
- Les arrêts se citent entre eux → un vrai réseau
- Un juriste fait ce travail à la main pendant des heures

**Outil cible**
Pour un argument donné, retrouver automatiquement :
- les **fondements textuels** pertinents
- la **jurisprudence** applicable (y compris contraire)

**Utilisateurs** : avocats, magistrats, juristes d'entreprise, particuliers
**Gain** : temps · exhaustivité · traçabilité

---

## Slide 3 — Problématiques

**Problématique métier**
- Recherche juridique = raisonnement multi-saut (article → arrêt → article cité → arrêt qui l'interprète)
- Le RAG vectoriel classique rate ces chaînes et ne restitue pas le raisonnement

**Enjeux transverses**
- **Traçabilité** : chaque réponse adossée à des sources citables
- **Transparence** : savoir *pourquoi* tel article/arrêt est remonté
- **Compréhension du raisonnement** : exposer la chaîne (fondement → JP → application)
- **Gestion des contradictions** : arrêts divergents, revirements, hiérarchie des juridictions

**Problématique technique**
- Construire un KG juridique FR depuis Judilibre/Légifrance
- Garantir la qualité d'extraction LLM (hallucinations = inacceptables)
- Interroger le graphe en préservant traçabilité et transparence
- **Maintenabilité** : ajouter de nouveaux arrêts/articles sans reconstruire tout
  - piste : mises à jour incrémentales (LightRAG-like)
- **Temporalité** : chaque entité a une dimension temporelle
  - articles : création, modification, abrogation (versions successives)
  - arrêts : date, juridiction, sens (confirmation, cassation, revirement)
  - besoin : requêter "le droit tel qu'il était à la date X" + suivre les revirements

**Problématique scientifique**
- *"Dans quelle mesure une architecture GraphRAG hybride peut-elle améliorer la recherche juridique FR en termes de factualité, traçabilité et explicabilité du raisonnement ?"*

> [!warning] Spécificité juridique
> Le droit n'est pas un corpus figé : un KG juridique sans versioning temporel ni update incrémental devient obsolète en quelques mois.

> [!tip] Punchline
> En droit, la bonne réponse mal justifiée = mauvaise réponse.

---

## Slide 4 — État de l'art : cartographie

- Visuel central : mindmap avec 6-7 branches
  - KG juridiques (Belikov-Raoult, D'Amato, LYNX…)
  - GraphRAG (Edge/Microsoft, Zhang, Peng, LightRAG, HippoRAG, KAG, StructRAG)
  - Construction KG avec LLM (Bian survey…)
  - NLP juridique & NER FR (JuriBERT, CamemBERT-Legal)
  - Ontologies juridiques (ELI, AKN, LKIF, ECLI)
  - Données FR (Judilibre, Légifrance, Dalloz)
  - Benchmarks & évaluation
- Chiffre en coin : 50 sources cataloguées, X fichées
- Message : balayage large avant positionnement

---

## Slide 5 — GraphRAG : concepts clés

**A. Les 3 types de Knowledge Graph** (Zhang 2025, Fig. 2)
- **Open-domain** : DBpedia, YAGO, Wikidata — larges, peu spécialisés
- **Domain-specific** : UMLS (médical), LYNX (juridique EU) — ontologie experte, coût élevé
- **Hybrid / corpus-based** : Microsoft GraphRAG, LightRAG, HippoRAG — extraits par LLM depuis le texte
- → Notre choix : hybride adossé à une ontologie juridique légère

**B. Les 3 phases d'un pipeline GraphRAG** (Zhang 2025, Fig. 3)
1. **Knowledge Organization** (G-Indexing) : chunking, extraction entités/relations, construction du graphe, communautés
2. **Knowledge Retrieval** (G-Retrieval) : matching requête → graphe, stratégies (sémantique, logique, GNN, LLM…)
3. **Knowledge Integration** (G-Generation) : injection dans le LLM (in-context, fine-tuning, refinement)

**C. Où chaque papier pivot se situe**
- Edge 2024 (Microsoft GraphRAG) → pipeline complet, focus phase 1 + map-reduce phase 3
- Zhang 2025 → taxonomie, couvre les 3 phases
- Belikov-Raoult 2025 → phase 1 appliquée à Cassation FR
- D'Amato 2025 → phase 1 avec ontologie ciblée (CEDH)

> [!info] Message
> Chaque papier répond à une partie du problème — aucun ne couvre tout le pipeline pour le droit français.

---

## Slide 6a — Microsoft GraphRAG + LightRAG

**Edge et al. 2024 — Microsoft GraphRAG**
- Idée : KG extrait par LLM + résumés de communautés Leiden → QFS map-reduce
- Apports : pipeline complet, gleanings, hiérarchie de communautés, LLM-as-judge
- Limite : pas de métrique factualité/traçabilité (bloquant pour juridique)
- Couvre : phases 1 + 3 · Type : hybride

**LightRAG (Guo et al. 2024)**
- Idée : dual-level retrieval (local entités + global relations) sur KG extrait
- Apports : incrémentalité (mise à jour sans rebuild), coût réduit vs GraphRAG
- Limite : évaluation sur QA générique, pas juridique
- Couvre : phases 1 + 2 · Type : hybride

---

## Slide 6b — KAG + StructRAG

**KAG — Knowledge Augmented Generation (Liang et al. 2024)**
- Idée : aligner KG et texte via une couche sémantique mutuelle, raisonnement logique-symbolique
- Apports : traçabilité native, décomposition de requête, bon sur domaines pros
- Intérêt pour nous : le plus proche d'un besoin "réponse justifiée" → axe traçabilité
- Couvre : phases 1 + 2 + 3 · Type : hybride + domaine

**StructRAG (Li et al. 2024)** *(à lire)*
- Idée : choisir dynamiquement la structure (table, graphe, arbre…) selon la requête
- Apports annoncés : meilleur sur tâches à raisonnement complexe
- À vérifier : applicabilité au juridique, coût d'inférence
- Couvre : phases 1 + 2 · Type : hybride adaptatif

---

## Slide 6c — Surveys & taxonomies

**Zhang et al. 2025 — GraphRAG Survey for Customized LLMs**
- Taxonomie 3 phases, 3 types de KG, Table 1 des méthodes de retrieval
- Rôle chez nous : boussole — classe chaque technique, aide à choisir

**(optionnel) Peng et al. 2024 — GraphRAG Survey ACM**
- Couverture plus générale, focus applications
- Complément historique

---

## Slide 6d — KG juridiques

**Belikov & Raoult 2025 — KG Cassation**
- Idée : KG de la Cour de cassation FR, extraction depuis arrêts
- Apports : preuve de faisabilité côté FR, schéma d'entités
- Limite : périmètre restreint, pas de GraphRAG dessus

**D'Amato 2025 — KG jurisprudence CEDH**
- Idée : KG thématique (violence faites aux femmes) avec ontologie dédiée
- Apports : méthodo d'ontologie ciblée, validation experte
- Limite : corpus monolingue anglais, domaine étroit

---

## Slide 7 — Points saillants transversaux

Tableau de synthèse :

| Axe | GraphRAG | LightRAG | KAG | Belikov | D'Amato |
|---|---|---|---|---|---|
| Construction LLM | ✅ | ✅ | ✅ | partiel | ontologie manuelle |
| Communautés | ✅ | — | — | — | — |
| Traçabilité native | — | — | ✅ | — | ✅ |
| Domaine juridique | — | — | — | ✅ FR | ✅ EN |
| Multi-saut | partiel | ✅ | ✅ | — | — |

**Lecture** : aucun papier ne coche toutes les cases → opportunité.

---

## Slide 8 — Gaps identifiés

- **Gap 1** : très peu de KG juridiques FR ouverts et réutilisables
- **Gap 2** : GraphRAG peu évalué sur critères juridiquement pertinents (factualité, traçabilité, revirements)
- **Gap 3** : ontologies FR (ELI, AKN…) fragmentées, pas unifiées pour JP + articles
- **Gap 4** : benchmarks LLMs en droit FR limités (peu de tâches de raisonnement multi-saut)
- **Gap 5** : peu de travaux sur la temporalité / versioning dans les KG juridiques

> Chaque gap = une porte d'entrée pour la contribution.

---

## Slide 9 — Positionnement & contribution

**Proposition** : construire un KG juridique FR + GraphRAG évalué rigoureusement, avec benchmark des LLMs FR en droit.

**Axes de contribution**

1. **Analyse comparative de LLMs open-source sur tâches juridiques FR**
   - Modèles : Gemma 3, Kimi, MiniMax, Llama, Mistral, Qwen…
   - Deux régimes de comparaison :
     - Zero-shot / few-shot (état brut)
     - Fine-tuning léger (LoRA/QLoRA) sur corpus juridique FR → mesurer le *delta*
   - Tâches : QA juridique, NER, classification, raisonnement multi-saut
   - Benchmark : existant (LexGLUE, LegalBench) ou construit sur mesure

2. **Ontologie juridique adaptée**
   - Partir d'ELI / AKN / LKIF
   - Adapter au périmètre (articles + JP + arguments)
   - Intégrer la temporalité (valid time / transaction time)
   - Livrable : schéma OWL/SHACL ou schéma Neo4j documenté

3. **Construction du KG**
   - Pipeline d'extraction (NER + RE) sur Judilibre/Légifrance
   - Architecture hybride 3 niveaux (entités / communautés / chunks)
   - Validation qualité (échantillon annoté)
   - Support des mises à jour incrémentales

4. **Évaluation comparative de méthodes de retrieval**
   - Parmi : similarity-based, logical reasoning-based, GNN-based, LLM-based, multi-round, hybrid (Zhang 2025 Table 1)
   - Comparaison quantitative sur tâches juridiques réelles

5. **Benchmark de pertinence vs systèmes existants**
   - Baselines : RAG naïf, Doctrine.fr / Dalloz IA, LLM seul
   - Métriques : factualité, traçabilité des citations, recall des fondements, gestion des revirements
   - Construire le benchmark si inexistant

> [!important] Force de cette contribution
> Les 5 axes forment un pipeline cohérent : chaque brique publiable indépendamment, l'ensemble = mémoire solide.

---

## Slide 10 — Roadmap vue d'ensemble

Timeline 7 phases (horizontale, curseur "ici" sur P0→P1)

| Phase | Objet | Lien avec contribution |
|---|---|---|
| P0 — État de l'art | Cartographie, gaps | *en clôture* |
| P1 — Ontologie & données | ELI/AKN + Judilibre/Légifrance | Axe 2 |
| P2 — Construction KG | Extraction LLM, hybride 3 niveaux | Axe 3 |
| P3 — Benchmark LLMs FR | Gemma/Kimi/MiniMax/Llama… (zero-shot + fine-tuning) | Axe 1 |
| P4 — Data science | Relations, centralité, communautés | Axe 3 enrichi |
| P5 — GraphRAG & retrieval | Méthodes comparées + benchmark final | Axes 4 + 5 |
| P6 — Rédaction mémoire | — | — |

**Points visuels**
- P1 et P3 peuvent avancer en parallèle (données vs modèles)
- P5 = aboutissement → dépend de P2 et P4
- Jalons superviseur : fin P1 (périmètre figé), fin P2 (KG V1), fin P3 (best LLM choisi), fin P5 (résultats finaux)

---

## Slide 11 — Prochaines étapes (4-6 semaines)

- Finir les fiches GraphRAG + KG juridiques (P0 closing)
- Lire StructRAG, KAG, HippoRAG
- Décider périmètre : JP seule ou JP + articles
- Explorer Judilibre/Légifrance — récupérer un échantillon
- Choisir base ontologique (ELI + extension ? AKN ?)
- **Points d'arbitrage avec le superviseur** :
  - périmètre du KG
  - priorité des axes de contribution
  - accès cluster GPU (pour P3)

---

## Slide 12 — Annexes / backup

- Tableau complet des 50 sources
- Schémas détaillés (pipeline Edge 2024, taxonomie Zhang 2025)
- Exemples bruts Judilibre/Légifrance
- Modèle temporel envisagé (optionnel)
- Liens Obsidian / Git
- Glossaire juridique (si superviseur non-juriste)

---

## Notes de préparation

- [ ] Préparer les schémas Partie B slide 2 (argument juridique)
- [ ] Récupérer Figures 2, 3, 4 de Zhang 2025 + pipeline Edge 2024
- [ ] Construire la mindmap slide 4
- [ ] Finaliser le tableau comparatif slide 7
- [ ] Construire la timeline slide 10
- [ ] Préparer template Beamer LaTeX
