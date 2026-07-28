---
date: 2026-07-28
type: conception-presentation
status: valide
owner: papier
cible: superviseur
tags: [papier, cvpr, presentation, plan-detaille, questions-reviewer]
---

# Conception du deck — questions du reviewer et sections du papier

## Décision structurante

Le deck suit l'ordre des questions qu'un reviewer doit pouvoir résoudre, tout en affichant explicitement la section du futur papier qui porte chaque réponse. Cette architecture, dite **option C**, a été validée le 28 juillet 2026.

Le support reste un plan détaillé destiné au superviseur. Il ne transforme pas les résultats exploratoires en conclusions et ne préjuge pas des résultats confirmatoires transmis par la tâche A.

## Principe de lecture

Chaque bloc du deck comporte trois niveaux :

1. la question du reviewer, utilisée comme titre ou amorce ;
2. la réponse actuellement défendable pour le projet ;
3. la section du papier qui devra porter cette réponse.

Les formulations s'inspirent de la structure d'introduction proposée dans `paper_writing.pdf`, pages 29--30, et de la progression problème, solutions antérieures, idée et validation proposée dans `How-to-write-a-good-CVPR-submission.pdf`, notamment pages 15, 18 et 25--29.

## Architecture narrative

| Ordre | Question du reviewer | Section du papier montrée | Contenu du deck |
|---:|---|---|---|
| 1 | Quel papier sommes-nous en train de construire ? | Vue d'ensemble | Objectif du rendez-vous et carte questions-sections |
| 2 | Pourquoi le problème est-il important ? | Introduction — motivation | Besoin juridique : retrouver cadre normatif et application jurisprudentielle |
| 3 | Pourquoi le problème est-il difficile ? | Introduction — difficultés | Volumétrie, accès, hétérogénéité, deux espaces candidats et liens difficiles à exploiter |
| 4 | Pourquoi les solutions existantes sont-elles insuffisantes ? | Introduction — lacune et Related Work | Retrieval textuel, graph retrieval et ressources françaises ; manque précis à vérifier |
| 5 | Quelle est l'idée principale ? | Introduction — intuition et teaser | Exploiter conjointement citations et proximités sémantiques dans un graphe typé |
| 6 | Quelle question scientifique testons-nous ? | Introduction — question et portée | Question centrale, droit français, évaluation pénale et deux modalités séparées |
| 7 | Que revendiquons-nous exactement ? | Introduction — contributions | Benchmark français, étude du graph design et diagnostic LLM aval ; performance conditionnelle |
| 8 | Quelle tâche et quelles données permettent de tester l'idée ? | Task and Benchmark | Deux classements, golds séparés, provenance, filtrage strict et limites du benchmark |
| 9 | Comment le droit est-il représenté ? | Method — Graph Construction | Inventaire $G_0$--$G_7$, nœuds, relations, blocs, poids et normalisations |
| 10 | Comment les sources sont-elles classées ? | Method — Retrieval | Cosine, PPR et LightGCN à protocole égal |
| 11 | Quel rôle le LLM joue-t-il réellement ? | Method — LLM Reranking | Capacité contrôlée sous distracteurs puis pipeline end-to-end |
| 12 | Comment rendre les hypothèses falsifiables ? | Experiments — Protocol | Folds groupés, sélection sur validation, évaluation interne et future lockbox |
| 13 | Quelles comparaisons peuvent soutenir ou réfuter les affirmations ? | Experiments — Results | Métriques, baselines, ablations, tableaux et figures nécessaires |
| 14 | Que pourra-t-on honnêtement conclure ? | Discussion and Limitations | Causalité, golds, contamination de l'évaluation interne, portée et données |
| 15 | Quel message doit rester au lecteur ? | Conclusion, puis Abstract | Implication scientifique conditionnelle ; résumé et conclusion écrits en dernier |

## Séquence de slides proposée

Le deck passe de vingt à vingt-quatre slides. L'augmentation sert à séparer la motivation, la difficulté, la lacune et l'intuition, aujourd'hui condensées dans une seule slide.

1. Titre.
2. Comment lire ce plan : questions du reviewer vers sections du papier.
3. Sommaire des sections du futur papier.
4. Pourquoi ce problème est-il important ?
5. Pourquoi est-il difficile ?
6. Que proposent déjà les travaux existants ?
7. Quel manque précis reste ouvert ?
8. Quelle est notre idée principale ?
9. Quelle question scientifique testons-nous, et sur quelle portée ?
10. Quelles contributions voulons-nous défendre ?
11. Quelle tâche évaluons-nous ?
12. Comment le benchmark français est-il construit ?
13. Quels graphes comparons-nous ?
14. Comment formaliser le graphe typé ?
15. Comment le premier étage classe-t-il les sources ?
16. Le LLM sait-il sélectionner les golds sous distracteurs ?
17. Le signal graphe reste-t-il utile après reranking ?
18. Comment séparons-nous tuning, confirmation interne et lockbox ?
19. Quelles métriques, baselines et ablations sont nécessaires ?
20. Quels tableaux répondront à quelles affirmations ?
21. Quelles figures doivent raconter l'argument ?
22. Quelles limites accompagnent les résultats ?
23. Que pourra affirmer la conclusion, puis l'abstract ?
24. Quelles décisions restent à prendre avec le superviseur ?

## Règles de mise en forme

- Le titre principal de chaque slide est la question du reviewer.
- Un sous-titre court indique la section du papier, par exemple `Section 1 — Introduction`.
- Les slides de méthode vont du général au détaillé : overview, graphe, retrieval, puis LLM.
- Les slides de résultats commencent par l'affirmation testée, puis indiquent la comparaison et la preuve requise.
- Les graphiques historiques restent accompagnés de la mention `Exploratoire`.
- Les slides ne contiennent aucun chiffre non relié à un `experiment_id` validé par la tâche A.
- L'abstract et la conclusion sont montrés dans le plan, mais ne sont pas rédigés à ce stade.

## Contenus conservés et contenus transformés

Les slides actuelles sur la tâche, le benchmark, les graphes, les méthodes, le protocole, les métriques, les résultats et les limites sont conservées sur le fond. Les changements portent sur leurs titres, leurs amorces et leur ordre.

L'ouverture est transformée plus fortement : l'actuelle slide d'introduction est décomposée en motivation, difficulté, lacune et intuition. La related work est divisée entre familles existantes et manque précis. Une nouvelle slide finale explicite ce que la conclusion et l'abstract pourront contenir une fois les preuves reçues.

## Critères de réussite

- Un superviseur peut comprendre le fil scientifique en lisant uniquement les titres des slides.
- Chaque question renvoie sans ambiguïté à une section du futur papier.
- La progression mène du problème aux preuves, et non d'une liste de composants à une autre.
- Le deck reste utilisable comme plan détaillé de rédaction après la réunion.
- Les statuts scientifiques restent visibles et conformes aux fichiers de coordination.
