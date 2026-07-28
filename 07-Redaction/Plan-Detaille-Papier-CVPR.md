---
date: 2026-07-28
type: plan-redaction
status: en-construction-collaborative
owner: papier
cible: CVPR
tags: [papier, cvpr, plan-detaille, retrieval-juridique, knowledge-graph]
---

# Plan détaillé — Papier CVPR Legal Knowledge Graph

## Statut du document

Ce document est le plan de travail destiné à être validé avant toute rédaction majeure du manuscrit. Il distingue les choix validés des propositions encore à discuter.

Support de discussion de référence :

- source Beamer : `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.tex` ;
- version compilée : `01-Projet/presentations/Plan-Detaille-Papier-Legal-KG-2026-07-28.pdf`.

La version PowerPoint du 27 juillet est conservée comme artefact intermédiaire, mais n'est plus le support courant.

### Architecture de discussion validée

Le support suit l'ordre des questions qu'un reviewer doit pouvoir résoudre, puis indique la section du papier qui porte chaque réponse :

1. importance du problème ;
2. difficulté du problème ;
3. solutions existantes et manque précis ;
4. idée principale, question de recherche et contributions ;
5. tâche, benchmark et méthodes ;
6. protocole, comparaisons et preuves ;
7. résultats, figures, limites et conclusions autorisées.

L'abstract et la conclusion apparaissent dans le plan, mais restent à écrire en dernier. Cette réorganisation narrative n'ajoute aucune preuve et ne change aucun statut expérimental.

- **Validé** : décision prise avec le superviseur du projet.
- **Partiellement validé** : objectif ou structure générale validé, détail restant à décider.
- **À valider** : proposition de travail, non encore figée.
- **Conditionnel** : contenu dépendant des résultats confirmatoires transmis par la tâche A.

## Cadrage scientifique validé

### Question de recherche centrale

> Dans quelle mesure la structure relationnelle du droit français, en particulier les citations entre jurisprudences et articles, améliore-t-elle la recherche conjointe des sources normatives et jurisprudentielles pertinentes pour une question juridique, au-delà d'un retrieval fondé uniquement sur la similarité sémantique ?

### Portée

- Le cadre méthodologique vise le retrieval en droit français.
- Le benchmark actuel, l'évaluation et les conclusions empiriques portent uniquement sur le droit pénal français.
- La généralisation aux autres branches du droit français n'est pas démontrée.
- Le papier traite du retrieval de sources, pas de la génération autonome d'une réponse ou d'un raisonnement juridique complet.

### Contributions candidates

1. Un benchmark de droit pénal français comportant deux tâches liées : retrieval d'articles et retrieval de décisions.
2. Une étude systématique des signaux citationnels et sémantiques et des méthodes qui les exploitent.
3. Un cadre de construction et d'évaluation de graphes juridiques hybrides, accompagné d'architectures concrètes contrôlant types de relations, poids et normalisation.
4. Une conclusion méthodologique conditionnelle sur l'utilité du typage et de la pondération, dont la force dépendra des résultats confirmatoires.

## Macrostructure validée

| Section | Titre de travail | Statut |
|---|---|---|
| 1 | Introduction | Validé |
| 2 | Related Work | Validé |
| 3 | Task and French Criminal-Law Benchmark | Validé |
| 4 | Legal Graph Design Space | Partiellement validé |
| 5 | Two-Stage Retrieval and Reranking Methods | Architecture validée, protocole à préciser |
| 6 | Experimental Protocol | À valider |
| 7 | Results and Ablations | À valider, résultats conditionnels |
| 8 | Analysis, Limitations and Threats to Validity | À valider |
| 9 | Conclusion | À valider en dernier |

## 1. Introduction — Validé

### Objectif

Motiver le besoin, traduire ce besoin en tâche scientifique, identifier le manque, présenter l'hypothèse relationnelle et annoncer les contributions sans anticiper les résultats.

### Progression paragraphe par paragraphe

1. **Besoin juridique** : répondre à une question suppose d'identifier le cadre normatif et son application jurisprudentielle.
2. **Difficulté** : décalage lexical, citations clairsemées, annotations incomplètes et exploitation limitée des ressources françaises ouvertes.
3. **Tâche scientifique** : retrouver séparément articles et décisions à partir d'une question en langage naturel.
4. **Hypothèse** : la structure citationnelle peut fournir un signal complémentaire à la similarité sémantique.
5. **Réponse proposée** : benchmark pénal français, espace structuré de graphes et architectures hybrides typées.
6. **Évaluation** : comparaison contrôlée des signaux et méthodes ; aucun chiffre avant validation par la tâche A.
7. **Contributions** : liste concise, avec toute supériorité empirique explicitement conditionnelle.

### Terminologie

- « retrouver des sources juridiques pertinentes », pas « produire la bonne réponse juridique » ;
- « sources normatives » ou articles ;
- « sources jurisprudentielles » ou décisions ;
- « décisions de jurisprudence » ou « décisions judiciaires » pour les documents individuels ;
- « similarité sémantique par embeddings » pour la baseline ; un graphe sémantique est une construction différente.

### Preuves nécessaires

- sources sur le besoin professionnel et les données françaises ouvertes ;
- audit bibliographique du manque scientifique ;
- définition et provenance du benchmark ;
- résultats confirmatoires avant toute phrase de supériorité.

## 2. Related Work — Validé

### Objectif

Faire émerger le positionnement par comparaison vérifiable avec les travaux les plus proches, sans affirmer prématurément une nouveauté absolue.

### 2.1 Legal information retrieval and benchmarks

- français juridique et retrieval d'articles : JuriBERT, BSARD et Finding the Law ;
- retrieval ou sélection de décisions : CaseHOLD, CaseLink, LePaRD et CLERC ;
- benchmarks juridiques plus généraux : LexGLUE, LEXTREME, LegalBench ;
- limites pour la tâche conjointe articles--décisions étudiée ici.

### 2.2 Graph-based legal retrieval

- graphes de citations ;
- recommandation de cas similaires ;
- GNN et apprentissage inductif, notamment Finding the Law et CaseLink ;
- réseau de citations et prédiction de liens, notamment LeCNet ;
- distinction entre retrieval, prédiction de citations et prédiction de jugement.

### 2.3 French legal resources and systems

- ressources ouvertes françaises ;
- Open French Law RAG ;
- KG de pourvois pénaux de la Cour de cassation par Belikov et Raoult ;
- périmètres et protocoles d'évaluation existants.

### 2.4 Positioning

Le gap candidat est limité à l'absence identifiée d'un benchmark ouvert en droit français réunissant une même question, deux cibles Articles--jurisprudences et une comparaison contrôlée des citations et proximités sémantiques. Il reste à consolider par la bibliographie primaire avant de figer `C008`.

Tableau comparatif selon : juridiction/langue, questions naturelles, articles, décisions, structure citationnelle, retrieval conjoint, benchmark public ou protocole reproductible.

### Preuves nécessaires

- lecture des sources primaires, en priorité BSARD et son extension GNN, CaseLink, Open French Law RAG et le KG français de la Cour de cassation ;
- vérification des tâches, datasets, métriques, licences et disponibilités ;
- validation ou réduction de l'affirmation de positionnement `C008`.

## 3. Task and French Criminal-Law Benchmark — Validé

### Objectif

Définir les sorties attendues, les composants du benchmark et leur construction avant d'introduire les graphes ou les méthodes.

### 3.1 Task definition

Pour une question (q), produire deux classements :

\[
q \mapsto R_A^K(q), \qquad q \mapsto R_J^K(q).
\]

- (R_A^K(q)) classe les articles ;
- (R_J^K(q)) classe les décisions ;
- espaces candidats, gold sets, métriques et résultats restent séparés ;
- aucun score unique ne fusionne les deux modalités.

### 3.2 Benchmark components

1. Dataset question--gold.
2. Corpus d'articles.
3. Corpus de décisions.

Pour chaque composant : source, version, période, périmètre, licence et disponibilité.

### 3.3 Dataset construction

- origine des documents servant à générer les questions ;
- génération et augmentation ;
- extraction et normalisation des références ;
- résolution des articles et décisions ;
- filtre `retrievable_strict` ;
- inventaire des exclusions et motifs.

### 3.4 Splits and evaluation roles

- apprentissage et validation croisée sur `train_augmented_retrievable_strict` ;
- évaluation interne sur `eval_rich_retrievable_strict` ;
- groupement des provenances et reformulations ;
- évaluation interne déjà consultée, donc absence de lockbox finale.

Les détails algorithmiques de `grouped_v2` restent en section 6.

### 3.5 Dataset statistics

Tableau compact : questions, documents sources, types, granularités, distributions des golds et tailles des espaces candidats. Chaque chiffre doit pointer vers un artefact versionné.

### 3.6 Scope and annotation limitations

- droit pénal uniquement ;
- questions générées ou augmentées ;
- gold potentiellement incomplet ;
- effet du filtre de retrouvabilité ;
- évaluation interne déjà consultée.

### Figure candidate

Pipeline reliant composants, construction des golds, filtre strict et splits. À arbitrer avec le programme global de figures.

## 4. Legal Graph Design Space — Partiellement validé

### Objectif

Présenter toutes les familles de graphes avant de détailler leur construction et leur rôle expérimental. Séparer clairement architecture, hyperparamètres et statut scientifique.

### 4.1 Overview of graph families — Validé dans son principe

Le début de section contient un tableau synthétique :

| Famille | Construction résumée | Relations principales | Poids / normalisation | Rôle expérimental |
|---|---|---|---|---|
| G0 | Graphe de citations initial | décision--article | structure source | référence brute |
| G1 | G0 sans articles isolés ou non en vigueur, puis décisions isolées | décision--article | structure source | contrôle citationnel nettoyé |
| G2 | G1 sans les 14 articles de plus haut degré | décision--article | structure source | effet des hubs |
| G3 | G2 sans une liste fixe d'articles procéduraux | décision--article | structure source | citation plus sélective |
| G4 | Graphe kNN mixte sur les nœuds G1 | AA, JJ et AJ | cosine, symétrisation par maximum | contrôle sémantique |
| G5 | Combinaison G1 et G4 | citation + tous blocs sémantiques | maximum arête par arête | union naïve |
| G6 | G1 plus blocs sémantiques sélectionnés | AA, JJ, AJ ou combinaisons | cosine, normalisation symétrique par bloc | effet du typage |
| G6U | Variante uniforme de G6 | blocs sélectionnés | arêtes binaires uniformes | contrôle de poids |
| G7 | G1 plus un bloc sémantique sélectionné | AA ou JJ dans la campagne principale | arêtes binaires, poids par famille | architecture pondérée |

Les formulations exactes seront vérifiées contre les métadonnées avant diffusion.

### 4.2 Formal graph schema — À valider

- ensembles de nœuds Articles et Décisions ;
- matrice bipartite de citations ;
- représentation carrée symétrique ;
- blocs de relations AA, JJ et AJ.

### 4.3 Graph construction — À valider

- filtrages G1--G3 ;
- construction kNN de G4 ;
- combinaison par maximum de G5 ;
- sélection et normalisation des blocs G6/G6U ;
- pondération explicite G7.

### 4.4 Concrete graph instances — Validé dans son principe

Un second tableau, en fin de section ou en supplément, inventorie chaque artefact concret : identifiant, (k), blocs, poids, normalisation, `experiment_id` et statut scientifique.

### Contrainte causale

G6 et G7 changent simultanément poids des arêtes, normalisation et pondération par famille. Leur différence directe ne doit pas être attribuée à la seule pondération. Les comparaisons causales devront rester internes à des constructions appariées ou être décrites comme comparaisons de régimes.

## 5. Two-Stage Retrieval and Reranking Methods — Architecture validée, protocole à préciser

### Objectif validé

Définir une architecture en deux étages : un retriever produit un vivier de candidats, puis un LLM reranke ce vivier pour construire le top-10 final. Distinguer la couverture du vivier de la qualité du reranking.

Le retrieval graphe constitue la contribution méthodologique principale. Le LLM reranker est le second étage du système final visé et sert à tester l'utilité aval du meilleur retriever sélectionné sur validation. Ses résultats resteront conditionnels à une nouvelle expérience conforme au protocole confirmatoire.

### 5.1 Stage 1 — Candidate retrieval

- Similarité cosine : baseline sans graphe.
- Personalized PageRank : propagation non apprise sur un graphe.
- LightGCN : propagation et apprentissage sur un graphe.
- Les stratégies de negative sampling sont des composants ou ablations de LightGCN, pas des méthodes de retrieval indépendantes.

Chaque méthode produit un classement séparé d'articles et de décisions ainsi qu'un vivier top-(K).

### 5.2 Stage 2 — LLM reranking

Le LLM reçoit un vivier de candidats et retourne dix références. Sa capacité de sélection et son utilisation dans le pipeline réel seront évaluées séparément.

#### 5.2.1 Controlled context-capacity study

Pour chaque question retenue, toutes les références gold annotées sont maintenues dans le vivier. Des distracteurs sont ajoutés progressivement pour faire varier la taille du vivier de (K=10) à (K=100). Les viviers doivent être imbriqués afin que le passage d'un (K) au suivant ajoute des distracteurs sans remplacer les candidats déjà présents.

Cette expérience mesure la probabilité que le LLM restitue les golds sachant qu'ils sont disponibles. Elle isole donc la robustesse du reranker à la taille du contexte et au nombre de distracteurs ; elle ne mesure pas la qualité du retriever.

Les distracteurs seront issus du classement d'un retriever afin de constituer des négatifs réalistes et difficiles. Les golds seront exclus du réservoir de distracteurs puis réinjectés dans chaque vivier. L'ordre présenté au LLM sera randomisé avec des seeds contrôlés, sans exposer le rang ni le score du retriever. La grille exacte de (K), la représentation textuelle, le budget de contexte et le nombre de répétitions restent à fixer.

#### 5.2.2 End-to-end retrieval and reranking

Le retriever produit réellement le vivier top-(K), sans injection de gold. Le LLM reranke ensuite ce vivier pour construire le top-10 final.

Comparaisons nécessaires :

1. retriever seul ;
2. LLM + baseline cosine ;
3. LLM + meilleur retriever graphe sélectionné sans utiliser l'évaluation interne.

### 5.3 Diagnostic decomposition

- étude contrôlée : rappel des golds par le LLM conditionnellement à leur présence garantie, en fonction de (K) ;
- pipeline réel : couverture du gold dans le vivier du retriever à chaque (K) ;
- pipeline réel : performance du top-10 après reranking ;
- perte entre l'oracle du vivier et la sortie du LLM ;
- Articles et JP toujours reportés séparément.

### 5.4 Scientific status

- Les sweeps LLM+RAG historiques restent exploratoires.
- L'application du LLM au meilleur retriever graphe constitue une expérience future proposée.
- La courbe contrôlée de capacité et l'évaluation end-to-end constituent deux expériences différentes ; aucune ne dispose encore de preuve confirmatoire.
- Toute sélection de la grille de (K), du prompt ou du modèle doit être faite sur validation, avant l'évaluation interne.
- M3 reste une méthode d'évaluation et sera décrite en section 6.

### Décisions restantes

- grille exacte de (K) entre 10 et 100 ;
- représentation des candidats et gestion du budget de contexte ;
- protocole de sélection du prompt et du modèle ;
- statut de LightGCN non entraîné ;
- répétitions nécessaires pour la variance du LLM.

## 6. Experimental Protocol — À valider

### Objectif proposé

Garantir comparabilité, absence de sélection sur l'évaluation interne et traçabilité de chaque résultat.

### Structure candidate

1. Folds `grouped_v2` et audit des fuites.
2. Métriques Articles et JP.
3. Sélection fold-first et couverture complète.
4. Grilles et budgets communs.
5. Sélection d'epoch et replay LightGCN figé.
6. Évaluation interne et future lockbox.
7. Deltas appariés, intervalles et seeds.
8. Manifestes, hashes et reproductibilité.

## 7. Results and Ablations — À valider, conditionnel

### Objectif proposé

Répondre aux affirmations dans un ordre causal et non chronologique.

### Structure candidate

1. Contrôles cosine et citationnels.
2. Progression G0--G7 avec statuts scientifiques visibles.
3. Résultats confirmatoires G1/G6/G7.
4. Ablations par type de relation.
5. Ablations de poids à construction constante.
6. Compromis Articles/JP.
7. Résultats négatifs et analyses exploratoires clairement séparés.

### Règle

Aucun chiffre n'entre dans cette section sans `experiment_id` et verdict transmis par la tâche A.

## 8. Analysis, Limitations and Threats to Validity — À valider

### Structure candidate

1. Analyse de couverture et de profondeur du ranking.
2. Erreurs par type et granularité de question.
3. Annotation incomplète et quasi-positifs.
4. Périmètre pénal et transférabilité non testée.
5. Questions générées ou augmentées.
6. Sélection historique sur l'évaluation interne.
7. Absence actuelle de lockbox inédite.
8. Coût, reproductibilité et limites des espaces candidats.

## 9. Conclusion — À valider en dernier

La conclusion répondra uniquement à la question centrale avec les résultats confirmés. Elle rappellera la portée pénale de la preuve et n'introduira ni nouvelle expérience ni liste de travaux futurs.

## Programme provisoire de figures et tableaux

| Élément | Fonction | Statut |
|---|---|---|
| Figure 1 | Question → articles + décisions → graphe → deux rankings | À valider |
| Tableau 1 | Positionnement face aux travaux proches | Validé dans son principe |
| Figure 2 | Construction du benchmark et splits | À valider |
| Tableau 2 | Statistiques du benchmark | Validé dans son principe |
| Tableau 3 | Vue d'ensemble G0--G7 | Validé dans son principe |
| Tableau S1 | Inventaire exhaustif des instances de graphes | Validé dans son principe |
| Tableau 4 | Résultats principaux Articles/JP | Conditionnel |
| Figure 3 | Deltas appariés ou compromis Articles/JP | Conditionnel |
| Figure 4 | Analyse de couverture/profondeur | Conditionnel |

## Carte courante des affirmations

La source de vérité est `01-Projet/paper-control/REGISTRE-AFFIRMATIONS.csv`. Les affirmations principales sont : qualité du protocole, complémentarité citation/sémantique, utilité éventuelle du graphe hybride, effets de l'union naïve et du negative mining, contribution benchmark et contribution méthodologique.

## Décisions restantes avant rédaction

1. Rôle final du LLM/RAG et protocole de sélection du vivier.
2. Structure exacte du protocole et vocabulaire des métriques.
3. Ordre des résultats selon les preuves transmises par A.
4. Programme final de figures et tableaux.
5. Contenu des limites et menaces à la validité.
6. Titre, résumé, conclusions et formulation finale des contributions, en dernier.
