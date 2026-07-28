---
date: 2026-07-28
type: conception-presentation
status: valide
owner: papier
cible: superviseur
tags: [papier, cvpr, presentation, introduction, related-work, contributions]
---

# Conception — révision de l'introduction et de la Related Work

## Décision validée

L'ouverture du deck est développée pour mieux expliquer le besoin juridique, la difficulté du retrieval et le positionnement réel du projet. La Related Work passe d'une slide générique à trois slides fondées sur des travaux identifiés. Une slide supplémentaire relie explicitement le retrieval aux assistants juridiques fondés sur des LLM.

Le deck passe de vingt-quatre à vingt-sept slides. Les slides techniques situées après les contributions sont conservées et décalées de trois positions.

## Règles scientifiques

- Ne pas affirmer que la recherche de jurisprudence est inexistante ou globalement peu étudiée.
- Distinguer modèles de langue, benchmarks, méthodes de retrieval et construction de graphes.
- Présenter le manque comme un manque identifié dans le droit français et dans le retrieval conjoint Articles--jurisprudences.
- Ne pas présenter une amélioration de performance comme une contribution démontrée avant validation par la tâche A.
- Conserver la future lockbox dans la slide de protocole, pas dans la définition des contributions.
- Ne pas attribuer à un papier une capacité qu'il n'évalue pas.

## Nouvelle séquence des slides 3 à 13

### Slide 3 — Quelles sections devront répondre à ces questions ?

Conserver la carte des sections et supprimer entièrement l'encadré :

> L'abstract sera écrit en dernier, une fois les affirmations finales et leurs `experiment_id` connus.

La règle reste valide pour le travail rédactionnel, mais elle n'apporte rien à cette slide de sommaire.

### Slide 4 — Pourquoi ce problème est-il important ?

**Section : Introduction — Motivation**

Point de départ : un avocat, un autre professionnel du droit ou un particulier cherche à répondre à une question juridique.

Deux familles de sources sont nécessaires :

1. les **articles**, qui définissent les règles et le cadre normatif ;
2. les **jurisprudences**, qui montrent comment ces règles sont interprétées et appliquées dans des situations concrètes.

Message final : une réponse juridique correctement fondée suppose de retrouver les deux, et non seulement un texte sémantiquement proche.

### Slide 5 — Pourquoi ce problème est-il difficile ?

**Section : Introduction — Difficultés**

Présenter quatre difficultés :

1. **Expertise juridique** : vocabulaire spécialisé, qualifications et dépendance au contexte ;
2. **Échelle et cas particuliers** : grand volume de textes et de décisions, exceptions et configurations factuelles nombreuses ;
3. **Organisation du droit** : domaines, sous-domaines, hiérarchies, renvois, citations et évolutions ;
4. **Limites de la similarité** : deux documents peuvent employer des termes proches tout en défendant des solutions opposées ou appartenir à des régimes distincts.

Ne pas écrire qu'aucune base n'utilise de relations. Écrire plutôt qu'aucun benchmark français ouvert identifié n'évalue systématiquement l'apport de ces relations pour les deux cibles.

### Slide 6 — Pourquoi le retrieval est-il indispensable aux assistants juridiques ?

**Section : Introduction — Motivation système**

Chaîne argumentative :

1. les assistants destinés aux professionnels du droit utilisent de plus en plus des LLM ;
2. un LLM non correctement ancré peut produire une réponse ou une citation non fiable ;
3. cette erreur est particulièrement grave dans un contexte juridique ;
4. le système doit donc retrouver les sources pertinentes avant toute génération ;
5. le papier étudie d'abord le retrieval, puis la capacité du LLM à sélectionner les bonnes sources dans le vivier récupéré.

La slide ne prétend pas que le système produit déjà une réponse juridique complète. Elle présente le retrieval comme condition nécessaire du système final visé.

### Slide 7 — Que sait-on faire en français sur les textes juridiques et les articles ?

**Section : Related Work — Français et statutory retrieval**

| Travail | Apport | Limite par rapport à notre problème |
|---|---|---|
| JuriBERT, Douka et al. 2021 | Modèle de langue adapté au français juridique | Représentation linguistique, pas benchmark de retrieval |
| BSARD, Louis et Spanakis 2022 | Questions françaises vers articles belges et baselines de retrieval | Articles seulement ; droit belge |
| Finding the Law, Louis et al. 2023 | GNN exploitant la structure hiérarchique des lois pour le retrieval d'articles | Structure statutaire uniquement ; aucune jurisprudence |
| Open French Law RAG, Harvard LIL 2024 | Étude RAG sur un corpus de droit français et analyse des erreurs | Étude de cas, pas benchmark conjoint Articles--JP |

### Slide 8 — Que sait-on faire pour retrouver des jurisprudences ?

**Section : Related Work — Legal case retrieval**

| Travail | Apport | Limite par rapport à notre problème |
|---|---|---|
| CaseHOLD, Zheng et al. 2021 | Sélection du holding pertinent parmi des candidats | QCM de compréhension, pas retrieval libre d'une jurisprudence française |
| CaseLink, Tang et al. 2024 | Graphe inductif combinant relations sémantiques et qualifications pour le case retrieval | Autre système juridique ; cas vers cas seulement |
| LePaRD, Mahari et al. 2024 | Retrieval de passages issus de précédents américains | Droit américain ; passages jurisprudentiels uniquement |
| CLERC, Hou et al. 2025 | Retrieval de décisions puis génération d'une analyse fondée sur les citations | Droit américain ; n'évalue pas le couple articles français et jurisprudences françaises |

### Slide 9 — Que sait-on faire avec des graphes juridiques ?

**Section : Related Work — Graphes et réseaux de citations**

| Travail | Apport | Limite par rapport à notre problème |
|---|---|---|
| Finding the Law, Louis et al. 2023 | Structure hiérarchique de la législation dans un GNN | Graphe d'articles uniquement |
| CaseLink, Tang et al. 2024 | Graphe global pour le retrieval de cas | Une seule cible documentaire |
| LeCNet, Harde et al. 2025 | Réseau de citations judiciaires pour la prédiction de liens | Droit indien et tâche de link prediction |
| Belikov et Raoult 2025 | Construction d'un KG à partir de pourvois pénaux de la Cour de cassation | Construction de KG, pas benchmark de retrieval conjoint |

### Slide 10 — Quel manque précis reste à établir ?

**Sections : Introduction et Related Work — Gap**

Formulation candidate :

> Les travaux existants étudient séparément la représentation du français juridique, le retrieval d'articles, le retrieval de jurisprudences ou la construction de graphes. Nous n'avons pas identifié de benchmark ouvert en droit français évaluant conjointement, pour une même question juridique, la récupération des articles applicables et des jurisprudences pertinentes, tout en comparant systématiquement l'apport des citations et des relations sémantiques.

La formulation reste une affirmation de Related Work à consolider par la bibliographie primaire. Elle ne doit pas être présentée comme une nouveauté définitivement prouvée à ce stade.

### Slides 11 et 12 — Idée principale, puis question scientifique

Conserver leur contenu actuel en adaptant la transition :

- slide 11 : graphe juridique puis retrieval et LLM reranking ;
- slide 12 : question centrale et portée empirique.

### Slide 13 — Que revendiquons-nous exactement ?

**Section : Introduction — Contributions**

Remplacer les formulations actuelles par trois contributions directement présentables :

1. **Benchmark de droit pénal français à deux cibles** : un dataset d'entraînement, un dataset d'évaluation interne et un protocole reproductible pour retrouver séparément articles et jurisprudences à partir d'une même question ;
2. **Comparaison d'architectures de graphes juridiques** : graphes de citations, graphes de proximité sémantique et architectures hybrides avec relations uniformes, typées ou pondérées, construits à partir de sources ouvertes ;
3. **Évaluation d'un pipeline retrieval--LLM** : comparaison de retrievers, étude contrôlée de sélection sous distracteurs et évaluation end-to-end distinguant absence du gold dans le vivier et échec du reranker.

Supprimer :

- l'encadré `Performance conditionnelle` ;
- la décision `framework hybride typé` contre `étude contrôlée du graph design`.

Les remplacer par une phrase simple :

> Le papier revendique les ressources, le protocole et l'étude comparative ; les conclusions de performance seront formulées uniquement après validation des expériences correspondantes.

## Sources primaires à afficher dans les slides

- JuriBERT : <https://aclanthology.org/2021.nllp-1.9/>
- BSARD : <https://aclanthology.org/2022.acl-long.468/>
- Finding the Law : <https://aclanthology.org/2023.eacl-main.203/>
- CaseLink : <https://arxiv.org/abs/2403.17780>
- LePaRD : <https://aclanthology.org/2024.acl-long.532/>
- CLERC : <https://aclanthology.org/2025.findings-naacl.441/>
- LeCNet : <https://aclanthology.org/2025.justnlp-main.4/>
- Open French Law RAG : <https://lil.law.harvard.edu/open-french-law-rag/>
- Knowledge Graphs Construction from Criminal Court Appeals : <https://arxiv.org/abs/2501.14579>

## Critères de réussite

- La motivation part d'un besoin juridique concret avant de présenter la technologie.
- La complexité juridique ne se réduit pas au volume documentaire.
- La Related Work nomme des travaux précis et explique leur apport comme leur limite.
- Le gap reste limité au contexte français et au retrieval conjoint Articles--jurisprudences.
- Les trois contributions sont compréhensibles sans vocabulaire interne au projet.
- Aucun résultat non validé n'est présenté comme une amélioration acquise.
