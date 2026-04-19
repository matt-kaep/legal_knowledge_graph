---
tags: [article, benchmark, rag, droit-francais]
categorie: "Benchmarks"
titre_complet: "Open French Law RAG : A Retrieval-Augmented Generation Benchmark for French Law"
auteurs: "Harvard Library Innovation Lab (LIL)"
annee: 2024
type: "Article de blog / rapport technique"
venue: "Harvard Law School Library Innovation Lab"
url: "https://lil.law.harvard.edu/open-french-law-rag/"
doi: ""
pdf_local: ""
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# Open French Law RAG — Harvard LIL 2024

> [!info] Metadonnees
> **Auteurs** : Harvard Library Innovation Lab (LIL)
> **Annee** : 2024 | **Venue** : Harvard Law School
> **Type** : Article de blog / rapport technique
> **URL** : [Lien](https://lil.law.harvard.edu/open-french-law-rag/)
> **Dataset** : COLD French Law Dataset (841 761 articles)

## Resume

Benchmark empirique testant GPT-4 et Llama2-70B sur des questions juridiques françaises, avec et sans RAG, sur un corpus de 841k articles de loi. Thèse centrale : **les LLM + RAG ne sont pas prêts pour la recherche juridique française sans supervision humaine**. Le RAG réduit les hallucinations (57% → 32%) mais fait chuter la cohérence logique (88% → 70%), sans gain significatif d'exactitude factuelle.

## Contributions principales

1. Premier benchmark RAG appliqué spécifiquement au droit français
2. Pipeline reproductible (embeddings multilingues + ChromaDB) sur un corpus massif
3. Mise en évidence des limites structurelles du RAG vectoriel sur du raisonnement juridique

## Methodologie

### Donnees
- Source : **COLD French Law Dataset**
- Volume : 841 761 articles → 965 129 vecteurs
- Langue : Français (avec questions en anglais/français)

### Architecture / Pipeline
- **Embeddings** : `intfloat/multilingual-e5-large` (100 langues)
- **Vector store** : ChromaDB
- **Retrieval** : top-4 extraits par question
- **LLMs testés** : GPT-4 (fermé) vs Llama2-70B (open)
- **Benchmark** : 10 questions à difficulté croissante
- **Variantes** : 4 (EN/FR × avec/sans RAG) × 2 modèles = **80 réponses**
- **Température** : 0.0 (déterministe)

### Criteres d'evaluation
Cohérence, couverture, consistance, exactitude, pertinence contextuelle, fidélité aux sources, qualité de traduction, hallucinations (intrinsèques vs extrinsèques).

## Resultats cles

| Metrique | Avec RAG | Sans RAG |
|---|---|---|
| Exactitude factuelle | 15% | 5% |
| Hallucinations | 32.5% | 57.5% |
| Cohérence logique | 70% | 88% |
| Extraits pertinents remontés | 29% (46/160) | — |
| Réponses citant les extraits | 9/40 | — |

### Par modèle
- **GPT-4** : 20% exactes avec RAG (meilleur)
- **Llama2-70B** : **0% exactes** dans tous les scénarios

### Par langue
- Anglais > français (y compris sur du droit FR)
- Le RAG atténue l'écart mais ne le ferme pas

## Points forts

- Premier benchmark reproductible RAG + droit FR
- Pipeline technique détaillé et public
- Mesure séparée hallucinations intrinsèques vs extrinsèques
- Met en lumière le problème fondamental du retrieval vectoriel en juridique

## Limites

- **N=10 questions** : échantillon statistiquement faible
- **Articles de loi seulement** : aucune jurisprudence dans le corpus
- **Pas de graphe** ni de structure relationnelle exploitée
- Pas de ground truth annoté rigoureusement
- Pas de mesure de sensibilité / reproductibilité
- Évaluation essentiellement humaine (pas de LLM-as-judge, pas de métriques automatiques)

## Conclusions principales

1. **Potentiel mitigé** : le modèle récupère souvent des documents non pertinents (71% hors contexte)
2. **Échec sur le cœur du raisonnement juridique** : portée matérielle, géographique, temporelle des règles
3. **RAG ≠ fiabilité** : moins d'hallucinations, mais pas plus de justesse
4. **Asymétrie d'usage** : experts filtrent les erreurs, novices tombent dans les "convincing hallucinations"
5. **Impact sur la pratique** : questionnements sur l'équilibre IA / jugement humain

## Liens avec mon projet

> [!important] Pertinence pour le KG juridique francais
> Ce papier **légitime** directement l'approche GraphRAG pour le droit français : il démontre empiriquement que le RAG vectoriel plat **échoue précisément sur les dimensions où un KG excelle** (portée temporelle, liens entre articles, raisonnement multi-saut). Leur 71% d'extraits hors contexte = ce qu'un graphe structuré corrigerait. À citer en introduction du mémoire comme motivation.

### Ce que je peux reutiliser
- **COLD French Law Dataset** : corpus directement réutilisable pour le volet articles
- Pipeline d'évaluation : grille de critères (cohérence, fidélité, hallucinations)
- Distinction hallucinations intrinsèques vs extrinsèques
- Structure du protocole (EN/FR × avec/sans) — adaptable aux 4 baselines

### Ce que je dois adapter / dépasser
- Échelle : passer de 10 à 100-500 questions
- Corpus : **ajouter la jurisprudence** (leur lacune majeure)
- Retrieval : comparer vectoriel vs graphe structuré
- Modèles : inclure Gemma 3, Mistral, Qwen (pas seulement GPT-4/Llama2)
- Ajouter métriques automatiques : LLM-as-judge, sensibilité, reproductibilité
- Construire un ground truth annoté manuellement
- Mesurer spécifiquement la **traçabilité** et le **raisonnement multi-saut**

### Limites à exploiter comme contribution
| Harvard LIL | Notre angle |
|---|---|
| N=10 questions | Benchmark 100-500 questions |
| Articles seuls | JP + articles |
| Retrieval vectoriel | Retrieval graphe |
| Pas de traçabilité mesurée | Métrique traçabilité explicite |
| Pas de sensibilité/reproductibilité | Protocole ablations propre |

## Connexions

### Articles lies
- [[Edge-2024-Microsoft-GraphRAG]] — leur diagnostic légitime l'approche GraphRAG
- [[Belikov-Raoult-2025-KG-Cassation]] — complémentaire (eux font le KG, Harvard fait le benchmark)
- [[DAmato-2025-KG-Violence-Women-CEDH]] — méthodologie CQ-driven à combiner ici

### Concepts lies
- [[RAG vectoriel]]
- [[Hallucinations intrinseques vs extrinseques]]
- [[COLD French Law Dataset]]
- [[Benchmark juridique]]

### Questions soulevees
- Peut-on réutiliser le COLD Dataset tel quel, ou faut-il le fusionner avec Judilibre ?
- Quel seuil d'exactitude viser pour valider notre GraphRAG ?
- Comment reproduire leur protocole en y ajoutant notre métrique de traçabilité ?

## Citations cles

> "Models frequently retrieve irrelevant documents and contain inaccuracies."

> "The models fail to correctly determine the material, geographical, and temporal scope of legal rules — a fundamental skill for lawyers."

> "RAG reduces overall hallucinations but does not systematically increase accuracy."

## Notes personnelles

- Le papier est un **argument en or** pour justifier la thèse du mémoire : le RAG vectoriel ne suffit pas, il faut de la structure.
- La chute de cohérence logique (88% → 70%) avec RAG est contre-intuitive et mérite d'être reproduite dans notre benchmark — possible que l'injection d'extraits hors contexte désorganise le raisonnement du LLM.
- Leur échec sur la *portée temporelle* confirme l'importance du sous-axe **temporalité** de notre projet.
- Manque majeur de leur étude : aucune JP. C'est exactement là où la recherche juridique réelle se joue.
- À explorer : le repo GitHub associé pour voir si le pipeline est réutilisable tel quel.
