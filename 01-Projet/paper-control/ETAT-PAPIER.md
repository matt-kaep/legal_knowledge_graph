---
date: 2026-07-27
type: etat-projet
status: active
owner: papier
tags: [papier, cvpr, redaction]
---

# État B — Papier

## Objectif

Construire un manuscrit d'une rigueur maximale, dans lequel chaque méthode est définie précisément et chaque affirmation quantitative est rattachée à une preuve auditable.

## État courant

- Une V0 LaTeX existe dans `07-Redaction/papier-v0/`.
- Elle contient déjà une première description des datasets, graphes, méthodes, métriques, résultats exploratoires et limites.
- Sa structure et ses chiffres doivent être réaudités ; elle ne constitue pas encore le manuscrit confirmatoire.
- La question de recherche centrale est validée ; elle porte sur la valeur ajoutée de la structure relationnelle du droit français au-delà d'un retrieval sémantique par embeddings.
- Les graphes hybrides typés et pondérés restent une hypothèse méthodologique à tester, et non la conclusion promise du papier.

## Question de recherche centrale validée

> Dans quelle mesure la structure relationnelle du droit français, en particulier les citations entre jurisprudences et articles, améliore-t-elle la recherche conjointe des sources normatives et jurisprudentielles pertinentes pour une question juridique, au-delà d'un retrieval fondé uniquement sur la similarité sémantique ?

Le papier distingue deux sorties de retrieval : les articles décrivent le cadre normatif applicable ; les décisions de jurisprudence documentent son interprétation et son application. Il ne revendique pas, à ce stade, la génération autonome d'une réponse ou d'un raisonnement juridique complet.

## Contributions candidates validées dans leur principe

1. Un benchmark français pour évaluer le retrieval conjoint d'articles et de jurisprudences à partir de questions juridiques.
2. Une étude systématique des signaux de retrieval : similarité sémantique, citations, propagation et apprentissage sur graphes.
3. Un cadre de construction et d'évaluation de graphes hybrides typés qui contrôle séparément la nature et le poids des relations, ainsi qu'une architecture juridique concrète qui instancie ce cadre pour combiner citations jurisprudence--article et proximités sémantiques.

Cette hiérarchie combine une contribution de ressource et d'évaluation, une contribution empirique et une contribution méthodologique. L'introduction pourra les annoncer de manière concise, mais la section Données devra documenter la provenance, la construction, le périmètre et les droits d'usage du benchmark.

La nouveauté méthodologique et la supériorité empirique sont deux affirmations distinctes. Le cadre et son instanciation peuvent être décrits comme contributions une fois leur définition et leur positionnement face à la littérature établis. Toute affirmation selon laquelle l'architecture surpasse les contrôles restera conditionnée par les résultats confirmatoires.

## Macrostructure validée

Le plan suivra une progression problème et benchmark avant méthode :

1. Introduction.
2. Related Work.
3. Tâche de retrieval et benchmark français.
4. Cadre de graphes hybrides typés.
5. Méthodes de retrieval.
6. Protocole expérimental.
7. Résultats et ablations.
8. Analyse, limites et menaces à la validité.
9. Conclusion.

Cette structure présente d'abord les sorties attendues, les données et les règles d'évaluation. Le cadre méthodologique est ensuite introduit comme une réponse au problème défini, avant toute comparaison empirique.

## Introduction — contrat validé

### Objectif

Motiver le besoin de retrouver les sources juridiques pertinentes, traduire ce besoin en tâche scientifique, identifier le manque traité, présenter l'hypothèse relationnelle et annoncer les contributions sans anticiper les résultats.

### Progression argumentative

1. Besoin réel : répondre à une question juridique suppose d'identifier le cadre normatif et son application jurisprudentielle.
2. Difficulté : décalage lexical, citations clairsemées, annotations incomplètes et exploitation encore limitée des ressources françaises ouvertes.
3. Problème scientifique : retrouver séparément articles et décisions à partir d'une question en langage naturel.
4. Hypothèse centrale : la structure citationnelle peut fournir un signal complémentaire à la similarité sémantique.
5. Réponse proposée : benchmark français, cadre de graphes hybrides typés et architecture juridique concrète.
6. Évaluation : comparaison contrôlée des signaux et méthodes ; aucun résultat quantitatif avant transmission confirmatoire de la tâche A.
7. Contributions : liste concise des contributions validées, avec supériorité empirique conditionnelle.

### Terminologie

- Employer « retrouver des sources juridiques pertinentes », et non « produire la bonne réponse juridique ».
- Distinguer « sources normatives » ou articles et « sources jurisprudentielles » ou décisions.
- Employer « décisions de jurisprudence » ou « décisions judiciaires » lorsque le texte désigne des documents individuels.
- Présenter la similarité sémantique par embeddings comme baseline ; un graphe sémantique est une construction distincte.

### Preuves requises

- Sources sur le besoin professionnel et l'ouverture des données françaises.
- Audit de la littérature pour étayer le manque scientifique français.
- Définition et provenance du benchmark dans la section dédiée.
- Résultats confirmatoires transmis par A avant tout chiffre ou langage de supériorité.

## Related Work — contrat validé

### Objectif

Positionner le travail par proximité avec la tâche étudiée et faire émerger le manque scientifique à partir de comparaisons vérifiables, sans transformer l'inventaire actuel en affirmation prématurée de nouveauté.

### Structure

1. **Legal information retrieval and benchmarks** : retrieval d'articles, retrieval ou sélection de décisions, benchmarks juridiques généralistes, puis limites pour le retrieval conjoint visé.
2. **Graph-based legal retrieval** : graphes de citations, recommandation de cas similaires, GNN et apprentissage inductif ; distinction entre retrieval, prédiction de citation et prédiction de jugement.
3. **French legal resources and systems** : données ouvertes françaises, systèmes RAG et KG français existants, avec leurs périmètres d'évaluation.
4. **Positioning** : synthèse du manque traité et tableau comparatif des travaux les plus proches.

### Axes du tableau comparatif

- juridiction et langue ;
- questions en langage naturel ;
- retrieval d'articles ;
- retrieval de décisions ;
- structure citationnelle ;
- retrieval conjoint ;
- benchmark public ou protocole reproductible.

### Preuves requises

- Lecture des articles primaires les plus proches, en priorité BSARD et son extension GNN, CaseLink, Open French Law RAG et le KG français de la Cour de cassation.
- Vérification des tâches, datasets, métriques, licences et disponibilités réelles.
- Formulation de nouveauté proportionnée au tableau ; éviter toute affirmation absolue non démontrée.

## Tâche et benchmark — décision de portée validée

Le projet définit un benchmark français unique composé de deux tâches liées à partir des mêmes questions juridiques :

1. **Retrieval d'articles** : classer les sources normatives pertinentes.
2. **Retrieval de décisions** : classer les sources jurisprudentielles pertinentes.

Chaque tâche conserve son espace candidat, son gold set, sa métrique principale et ses résultats. Les performances seront présentées côte à côte, sans score global fusionné susceptible de masquer une régression sur une modalité. Le terme « retrieval conjoint » désigne donc le traitement coordonné des deux dimensions d'une même question, et non leur réduction à une métrique unique.

### Portée scientifique

Le cadre méthodologique est conçu pour le retrieval en droit français. Le benchmark actuel, son évaluation et toutes les conclusions empiriques sont limités au droit pénal français. La transférabilité aux autres branches constitue une hypothèse externe non testée et sera présentée comme telle dans les limites.

## Section 3 — Tâche et benchmark pénal français

### Objectif

Définir précisément les deux sorties de retrieval, présenter les composants du benchmark et leur provenance, expliquer sa construction, puis documenter sa portée et ses limites avant d'introduire toute méthode.

### Structure validée

1. **3.1 Task definition** : formalisation de $q \mapsto R_A^K(q)$ et $q \mapsto R_J^K(q)$, espaces candidats et gold sets séparés.
2. **3.2 Benchmark components** : dataset question--gold, corpus d'articles et corpus de décisions ; sources, versions et licences.
3. **3.3 Dataset construction** : origine documentaire, génération ou augmentation, extraction et normalisation des références, résolution des golds, filtrage `retrievable_strict` et inventaire des exclusions.
4. **3.4 Splits and evaluation roles** : apprentissage/validation croisée, évaluation interne, provenance des groupes et absence de lockbox finale ; les détails algorithmiques de `grouped_v2` restent en section 6.
5. **3.5 Dataset statistics** : questions, documents sources, types, granularités, distributions des golds et tailles des espaces candidats.
6. **3.6 Scope and annotation limitations** : périmètre pénal, nature générée ou augmentée des questions, gold incomplet, effet du filtre de retrouvabilité et évaluation interne déjà consultée.

### Figures et tableaux candidats

- Une figure de pipeline reliant composants, construction et splits, à décider avec l'ensemble du programme de figures.
- Un tableau compact de statistiques, dont chaque nombre devra être relié à un artefact de données versionné.

### Preuves requises

- Manifestes et versions exactes des datasets et espaces candidats.
- Provenance et droits d'usage des trois composants.
- Comptes avant/après filtrage et motifs d'exclusion.
- Statistiques auditées et hashes des splits.

## Section 4 — Vue complète des graphes validée

La section méthodologique commencera par un tableau présentant l'ensemble des familles G0--G7 et leurs spécificités. G6U sera distingué comme contrôle uniforme interne à la famille G6. Le tableau devra rendre visibles :

- la base citationnelle ou sémantique ;
- les types de relations actifs ;
- le mode de poids des arêtes ;
- la normalisation ;
- les principaux hyperparamètres structurels ;
- le rôle expérimental ;
- le statut scientifique des résultats disponibles.

La section détaillera ensuite les constructions nécessaires pour comprendre les comparaisons, tout en séparant les explorations historiques de la campagne confirmatoire.

Deux niveaux de tableau sont validés : une synthèse des familles G0--G7 au début de la section, puis un inventaire exhaustif des instances concrètes en fin de section ou en supplément.

## Section 5 — Architecture en deux étages validée, protocole à préciser

1. **Candidate retrieval** : cosine, PPR et LightGCN produisent un vivier top-$K$ séparé pour Articles et JP.
2. **LLM reranking** : le LLM reçoit le vivier du retriever sélectionné et retourne le top-10 final.

L'analyse distinguera désormais deux expériences. Une étude contrôlée maintiendra les golds dans des viviers imbriqués de taille $K=10$ à $K=100$ et ajoutera progressivement des distracteurs afin de mesurer la capacité propre du LLM sous augmentation du contexte. Une étude end-to-end utilisera les vrais top-$K$ du retriever, sans injection de gold, afin de mesurer successivement couverture du vivier et qualité du reranking. La grille exacte de $K$ et la construction des distracteurs ne sont pas encore figées.

Les sweeps historiques restent exploratoires ; ni l'étude contrôlée ni l'application au meilleur retriever graphe ne disposent encore d'une preuve confirmatoire.

Pour l'étude contrôlée, la construction des distracteurs est validée : négatifs difficiles issus d'un retriever, golds garantis dans chaque vivier imbriqué, rangs et scores masqués, ordre randomisé avec seeds contrôlés. La grille exacte de $K$, le format des candidats, le budget de contexte et les répétitions restent ouverts.

Le retrieval graphe est la contribution méthodologique principale. Le LLM reranker est validé comme second étage du système final visé, et non comme simple extension exploratoire. Toute affirmation sur son gain reste conditionnelle à une nouvelle expérience validée par la tâche A.

### Contrainte d'interprétation identifiée

G6 et G7 ne diffèrent pas uniquement par le rapport de poids : le mode de poids des arêtes et la normalisation changent également. Une comparaison directe G6--G7 ne peut donc pas être interprétée comme l'effet causal isolé de la pondération. La formulation scientifique de ces deux régimes reste à décider avant de figer les équations et les ablations.

## Sections écrites sans attendre les runs

1. Problème et motivation.
2. Construction et provenance des datasets.
3. Définitions formelles des graphes.
4. Définition des méthodes et équations.
5. Métriques et protocole confirmatoire.
6. Limites connues et menace de contamination de l'évaluation interne.

## Sections conditionnelles

- Résumé et contributions finales.
- Tableau principal de résultats.
- Supériorité éventuelle du graphe hybride typé.
- Conclusions quantitatives et recommandations finales.

## Prochaine sortie attendue

Utiliser la présentation de plan avec le superviseur, puis décider le nom et la portée exacte de la contribution méthodologique avant de figer la définition formelle du graphe typé.

## Présentation de travail

Le support de référence est désormais la présentation Beamer :

- source : `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex` ;
- PDF compilé : `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.pdf`.

Elle contient vingt-quatre slides organisées selon les questions successives du reviewer : importance, difficulté, solutions existantes, manque, idée, question scientifique, contributions, tâche, méthodes, preuves et limites. Chaque slide indique la section du futur papier qui porte la réponse. Elle reprend le format des présentations hebdomadaires, notamment la présentation Week 16 V2. Les visuels historiques sont explicitement marqués comme exemples structurels ou exploratoires. Les vingt-quatre pages ont été rendues et vérifiées visuellement sans débordement. La version PowerPoint du 27 juillet reste conservée comme artefact intermédiaire et n'est plus le support courant.

Cette réorganisation est strictement narrative : elle n'ajoute aucune preuve, ne modifie aucun chiffre et ne change aucun statut expérimental.

## Preuves reçues de l'assainissement

Voir `SYNC-ASSAINISSEMENT-VERS-PAPIER.md`.

## Dernière mise à jour

2026-07-28 — Support superviseur réorganisé en vingt-quatre slides selon les questions du reviewer et les sections correspondantes du papier ; compilation sans avertissement et vérification visuelle intégrale. Aucun résultat ni statut scientifique modifié. Architecture retrieval puis LLM validée, étude LLM contrôlée séparée de l'end-to-end, prochaines décisions ordonnées avant rédaction.
