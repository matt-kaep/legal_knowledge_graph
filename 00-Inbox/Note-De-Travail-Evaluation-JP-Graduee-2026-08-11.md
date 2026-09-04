---
date: 2026-08-11
type: question
status: implemente-a-executer
tags: [benchmark, jurisprudence, llm-judge, evaluation-graduee, g8, avocat]
---

# Note de travail — évaluation graduée des jurisprudences

> Note vivante de conception. Les catégories, métriques et seuils restent exploratoires tant que le protocole n'est pas validé et enregistré par la tâche A.

## Objectif

Compléter l'évaluation exacte des JP attendues par une évaluation graduée de chaque couple `(question, JP retournée)`. Pour chaque question, les dix premières JP de chaque méthode sont jugées individuellement, puis un échantillon stratifié des jugements LLM est contrôlé en aveugle par un avocat.

## Décisions de travail déjà convergentes

- Conserver le `Hit@10 exact` actuel comme mesure automatique, reproductible et étroite.
- Ajouter une voie distincte de pertinence juridique graduée ; elle ne remplace pas rétroactivement la Ground Truth.
- Juger un pool dédupliqué de couples `(question, JP)` sans révéler au juge la méthode, le rang, la Ground Truth ni une relation G8 éventuelle.
- Réutiliser un jugement identique lorsqu'un même couple apparaît dans plusieurs méthodes.
- Réutiliser exactement la carte Step1 déjà calculée pour G8 ; ne pas recalculer ni enrichir une nouvelle fiche juridique pour chaque JP.
- Donner au juge uniquement la question et cette carte juridique existante.
- Faire contrôler les jugements par un avocat sur un échantillon stratifié ; 100 couples constituent un pilote, pas encore une validation définitive.
- Seules les classes A et B rapportent des points. Les classes C, D et E valent zéro, mais restent conservées pour expliquer les erreurs et la composition des résultats de chaque méthode.
- Autoriser `non_jugeable` lorsque la carte Step1 ne contient pas assez d'information pour qualifier le couple ; cette sortie est distincte de E, ne rapporte aucun point et reste dans le dénominateur K fixe.

## Ce que M3 fait déjà

M3 juge déjà chaque couple `(question, JP)` avec trois classes `n2`, `n1`, `n0`, puis calcule une moyenne. Son état actuel reste exploratoire : le prompt JP reçoit une synthèse, les classes sont larges, les poids `2/1/0` sont conventionnels, le rang n'entre pas dans le score et aucune calibration avocat systématique n'est enregistrée.

## Ce que G8 donnait réellement au LLM

Le juge G8 ne recevait pas directement le texte intégral Judilibre. Il recevait deux cartes Step1 compactes contenant, pour chaque décision :

- `synthese_pour_avocat` ;
- `fondements_retenus` ;
- `cited_articles` ;
- `solution_resume` ;
- jusqu'à six entrées `arguments_parties`, chacune séparant `argument` et `reponse_juge`.

Le prompt G8 exigeait ensuite une `notion_juridique` précise, une `regle_ou_raisonnement_applique` concrète et un type parmi `same_legal_issue`, `same_rule_application` ou `no_link`.

L'audit humain ultérieur des rattrapages G7/G8 a, lui, utilisé les textes intégraux Judilibre. Il faut donc distinguer le contexte du jugement LLM brut G8 et la preuve en texte intégral de l'audit.

Cette même carte Step1 devient l'entrée JP figée de la nouvelle évaluation. Pour un couple à juger, le prompt contient donc seulement :

1. la question juridique ;
2. la carte Step1 de la JP candidate.

## Informations Judilibre disponibles

Sur les 880 décisions matérialisées pour l'audit G8 :

- texte intégral : 880/880 ;
- solution : 880/880 ;
- articles de code extraits : 880/880 ;
- zones structurées : 509/880 ;
- motivation : 508/880 ;
- dispositif : 508/880 ;
- moyens : 404/880 ;
- exposé des faits et de la procédure : 309/880 ;
- visas : 571/880 ;
- thèmes : 534/880 ;
- résumé : 426/880 ;
- décision attaquée : 557/880 ;
- titres et sommaires substantiels : 59/880.

La longueur du texte intégral est très variable : médiane de 5 678 caractères, 95e percentile de 21 841 caractères et maximum observé de 162 901 caractères. Ces textes ne sont pas ajoutés à l'entrée du juge LLM dans le protocole retenu. Ils restent disponibles comme preuve pour l'audit avocat.

## Entrée figée du juge

Le protocole ne construit pas de nouvelle représentation. Il réutilise les champs G8 suivants :

1. `synthese_pour_avocat` ;
2. `fondements_retenus` ;
3. `cited_articles` ;
4. `solution_resume` ;
5. `arguments_parties[{argument, reponse_juge}]`.

Distinction obligatoire : un moyen est un argument présenté au juge ; il ne devient pas une règle retenue tant que la motivation ou le dispositif ne le confirme pas.

## Taxonomie à préciser

Les noms génériques `directement pertinente`, `juridiquement utile` et `connexe` sont rejetés comme trop flous. La future classe doit être déduite de champs juridiques explicites, au minimum :

- question juridique effectivement traitée par la décision ;
- règle, condition, test ou raisonnement appliqué ;
- réponse donnée par le juge ;
- contribution exacte de cette réponse à la question évaluée ;
- présence éventuelle d'une simple proximité de procédure, de faits ou de vocabulaire.

Ces éléments sont des critères de raisonnement donnés dans le prompt, pas des champs que le LLM doit tous reproduire. La carte Step1 contient déjà l'analyse juridique de la JP ; la sortie du juge doit rester minimale pour éviter la répétition, les contradictions et les hallucinations secondaires.

Taxonomie de travail validée dans son principe :

- **A — Règle directement applicable à la question** : la JP applique une règle ou un raisonnement qui permet de répondre directement à la question.
- **B — Apport par distinction, exception ou limite** : la JP traite la même question juridique et apporte une nuance substantielle utile pour répondre.
- **C — Même question juridique, sans solution exploitable** : la JP rencontre le problème, mais ne fournit pas de réponse utilisable sur le fond.
- **D — Proximité factuelle ou procédurale seulement** : même contexte, faits ou procédure, sans règle utile pour répondre.
- **E — Aucun rapport juridique substantiel** : hors sujet ou proximité lexicale superficielle.

Les règles de décision détaillées et les cas frontières restent à définir. Seules A et B contribuent au score ; C, D et E sont des catégories diagnostiques de score nul.

## Sortie minimale du juge

Le LLM ne reconstruit pas une seconde fiche juridique. Il renvoie uniquement :

```json
{
  "classe": "A | B | C | D | E | non_jugeable",
  "justification": "Une phrase qui nomme la règle ou la solution concrète retenue dans la JP et explique en quoi elle répond, complète ou ne répond pas à la question."
}
```

L'arbre A--E guide le raisonnement dans le prompt, mais aucune chaîne de pensée ni liste de sous-champs n'est demandée. `non_jugeable` est réservé aux cartes qui ne contiennent objectivement pas assez d'information pour qualifier le couple. Il ne sert pas de classe d'hésitation entre A--E et il ne vaut pas E. Il ne rapporte aucun point, mais le résultat occupe bien l'une des K positions du ranking et reste donc dans le dénominateur fixe. Son taux est reporté séparément par méthode.

La justification ne doit jamais paraphraser la classe. Les formulations génériques comme « la décision applique directement la règle permettant de répondre à la question » sont interdites. Une justification valide donne la substance juridique décisive, par exemple : « La Cour juge que l'interdiction de dissimuler intégralement le visage constitue une restriction nécessaire à l'ordre public au sens de l'article 9 CEDH, ce qui répond à la compatibilité demandée. »

## Score normalisé

Les gains sont ramenés directement dans l'intervalle `[0, 1]` :

- A : `1` ;
- B : `0,5` ;
- C, D et E : `0` ;
- `non_jugeable` : `0` point, avec une distribution séparée de E.

Pour une question `q` et un ranking de profondeur fixe `K` :

```text
score_gradue@K(q) = somme(gain(classe_i), i = 1..K) / K
```

Pour `K=10` :

```text
score_gradue@10(q) = (nombre_A + 0,5 * nombre_B) / 10
```

Les nombres A et B ci-dessus comptent seulement la première occurrence de chaque JP. Si une même JP apparaît plusieurs fois pour une question, les répétitions suivantes restent dans les dix positions mais valent zéro : elles représentent des places de retrieval perdues, pas de nouvelles réponses juridiques.

Le score vaut donc toujours entre `0` et `1`. Les classes C, D, E et `non_jugeable`, ainsi que les positions manquantes si une méthode retourne moins de K JP, contribuent zéro au numérateur sans réduire le dénominateur.

Reporter obligatoirement en parallèle :

```text
taux_non_jugeable@K = nombre de sorties non_jugeable / nombre de couples top-K attendus
```

Le score global d'une méthode est la moyenne des `score_gradue@K(q)` sur toutes les questions. Le taux `non_jugeable` et la distribution A--E sont publiés séparément.

## Organisation du chantier

Le programme est organisé en trois macro-chantiers successifs.

### Chantier 1 — Construire et geler l'instrument d'évaluation

1. Finaliser le prompt A--E, la sortie minimale et les cas frontières.
2. Choisir le modèle juge et enregistrer sa version, le prompt et leurs hashes.
3. Calibrer le prompt uniquement sur des questions d'entraînement ou un échantillon de calibration séparé, jamais sur les 754 questions d'évaluation interne.
4. Geler avant le run complet : modèle, prompt, température, schéma JSON, `K`, gains et formule.
5. Vérifier la couverture des cartes Step1 pour toutes les JP à juger.

Pour avancer vite, le premier run juge uniquement les top-10 JP de G7 sur les 754 questions. Le cosine brut et les autres méthodes ne sont ajoutés qu'au chantier 3, avec le même juge déjà gelé, afin de diagnostiquer les gains et pertes de G7 sans élargir le contrôle avocat initial.

### Chantier 2 — Produire puis valider la nouvelle évaluation

1. Récupérer les top-K gelés de chaque méthode.
2. Construire l'union dédupliquée des couples `(question, JP)` et rattacher la carte Step1 G8 existante.
3. Cacher au juge la méthode, le rang, la Ground Truth et les relations G8.
4. Juger chaque couple unique une fois, avec seulement `question + carte Step1`.
5. Reconstituer pour chaque méthode le `score_gradue@K`, la distribution A--E et le taux `non_jugeable`.
6. Tirer 100 couples stratifiés et les faire annoter en aveugle par un avocat, avec accès à la décision complète pour contrôler la vérité juridique et pas seulement la cohérence de la carte.
7. Mesurer l'accord global, l'accord pondéré et les confusions entre classes, en particulier la précision de A et B.

Si le contrôle avocat conduit à changer le prompt, le premier échantillon devient un échantillon de calibration. Une nouvelle vérification indépendante est alors nécessaire avant d'utiliser la métrique dans le papier.

### Chantier 3 — Comprendre puis améliorer G7

Comparer, question par question, le cosine brut et G7 avec deux lectures complémentaires :

- résultat exact contre la Ground Truth historique ;
- résultat gradué contre A--E.

Cette matrice sépare notamment :

- JP exacte manquée mais alternative A/B retrouvée : Ground Truth probablement incomplète, retrieval juridiquement utile ;
- JP exacte retrouvée mais classée C/D/E : cas à réauditer côté benchmark ou juge ;
- A/B remontée par G7 mais pas par cosine : amélioration attribuable à la propagation, à confirmer ;
- A/B perdue entre cosine et G7 : dégradation causée par G7 ;
- sorties C/D dominantes : pollution procédurale, factuelle ou thématique ;
- absence d'A/B dans les deux méthodes : problème en amont de candidats ou limite intrinsèque des représentations.

Les relations G8 interviennent seulement après cette mesure pour expliquer les cas : lien LLM brut direct par règle, même question juridique, proximité procédurale ou absence de paire candidate. Les distances 2--3 restent des diagnostics structurels et ne deviennent pas des preuves de pertinence.

Une modification de G7 ou l'utilisation d'arêtes G8 finales constitue ensuite une expérience distincte : tuning sur train/CV, graphe gelé, puis évaluation indépendante. La nouvelle métrique ne doit pas servir à choisir des hyperparamètres sur les 754 questions internes.

## Logique d'ensemble

```text
Rankings gelés
  -> pool dédupliqué question--JP
  -> carte Step1 G8 existante
  -> juge LLM aveugle A--E
  -> score_gradue@K + distributions
  -> contrôle avocat
  -> comparaison cosine/G7
  -> diagnostic par G8
  -> nouvelle expérience d'amélioration de G7
```

## Questions ouvertes

- La taxonomie, le modèle initial, le score et les seuils du pilote sont figés dans E016 ; ils ne peuvent être modifiés qu'après calibration train et changement explicite de version/hash.
- Restent ouverts empiriquement : la couverture réelle des fiches Step1 sur les 7 540 positions, la distribution A–E, le score gradué G7 et l'accord avocat.
- Le diagnostic cosine/G8 et toute modification de G7 appartiennent au chantier 3, après les résultats E016 et le gate avocat.

## État d'implémentation au 11 août

- spécification : `docs/superpowers/specs/2026-08-11-g7-graded-jp-evaluation-lawyer-audit-design.md` ;
- plan : `docs/superpowers/plans/2026-08-11-g7-graded-jp-evaluation.md` ;
- scripts reproductibles : 74 à 80 ;
- expérience enregistrée : E016 ;
- préflight et préparation terminés : 7 540 positions eval, 7 487 couples uniques, zéro fiche manquante et 53 positions répétées ; bundle train-only de 30 questions/298 couples préparé ;
- étape suivante : exécuter et inspecter le pilote train-only sur GPU, puis lancer le jugement complet ;
- aucun score E016 produit à ce stade.

Trois tentatives Slurm ont échoué avant tout jugement : cache non accessible, révision Hugging Face non résolue, puis OOM CUDA après chargement du snapshot sur `nodemm02`. Le snapshot complet est maintenant pin ; la reprise attend le choix d'une partition GPU dédiée, de préférence `L40S` déjà validée par les runs G8.

## Liens

- [[Audit-Juridique-G7-G8-2026-08-10]]
- [[Vocabulaire-Commun-G7-G8]]
- [[handoff-M3-LLM-judge]]
