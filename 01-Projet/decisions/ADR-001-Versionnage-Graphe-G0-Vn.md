---
tags: [decision, architecture, gnn, benchmark]
statut: "proposee"
date_decision: 2026-06-08
remplace: ""
---

# ADR-001 : Versionnage du graphe (G0→Vn) et régimes de supervision pour l'évaluation des baselines

## Contexte

L'évaluation des baselines (cosine, PPR, LightGCN, …) du grand tableau global repose sur le graphe biparti Article↔JP (`graph_penal.npz`) et la cohorte de 971 questions. Une **vérification factuelle des données** (2026-06-08) révèle que le graphe actuel est **brut et bruité** :

- **87 821 articles**, dont seulement **31 357 (36 %) ont un texte résolu/embeddé** (`resolution_rate = 0,357`).
- Degré de citation : **médiane 0** — **59 945 articles (68 %) ne sont jamais cités** (degré 0).
- **41 824 articles (48 %) sont « doublement morts »** : ni texte ni citation → init aléatoire jamais mise à jour, inertes, jamais retrouvables.
- Graphe **binaire** (`data` ne contient que des `1`) : la fréquence de citation est perdue.
- **Supervision JP quasi absente du train** : sur 978 questions à GT-JP, **971 sont dans la cohorte d'éval** et seulement **~7 dans le pool d'entraînement**.

**Reframing retenu** (proposé par M. Kaeppelin) : le graphe actuel est une **version G0** (baseline brute). L'objectif n'est pas que G0 soit bon, mais de **faire évoluer le graphe G0 → V1 → … → Vn** par nettoyages successifs, et de **mesurer le Δ métrique** à chaque version — la courbe ascendante *prouve* la valeur du nettoyage. Deux questions de design en découlent.

## Options considerees

### Dimension 1 — Versionnage du graphe (la décision principale)

**Option A — One-shot.** Nettoyer le graphe « du mieux possible » puis évaluer une fois.
- Avantages : simple, un seul run.
- Inconvénients : ne mesure pas *quel* nettoyage apporte quoi ; pas de récit expérimental ; non reproductible.

**Option B — Série versionnée G0→Vn (retenue).** Geler G0, puis publier une suite de versions, chacune = une opération de nettoyage + son Δ métrique.
- Avantages : chaque nettoyage devient une hypothèse testable ; courbe monotone = contribution démontrée ; reproductible ; cadre tout le Chantier 2.
- Inconvénients : discipline d'éval stricte requise (sinon les Δ sont ininterprétables) ; plusieurs runs.

### Dimension 2 — Régime de supervision (l'expérience « voir la différence »)

Les meilleures données (questions à GT **JP + article**) sont **entièrement dans la cohorte d'éval** → le train n'a aucun signal JP. Faut-il en injecter au train ?

**Option C — Held-out strict.** La cohorte 971 reste 100 % hors entraînement (anti-leak pur).
- Avantages : comparable aux autres baselines, zéro fuite, c'est **le tableau principal**.
- Inconvénients : aucune supervision JP ; n'exploite pas la meilleure donnée.

**Option D — Augmentation diagnostique (retenue, en table séparée).** Injecter une partie des questions de qualité (JP+article) dans le train pour **mesurer la différence**, en table à côté.
- Avantages : seul moyen d'avoir du signal JP au train ; quantifie la valeur de la donnée de qualité.
- Inconvénients : **non comparable** au tableau principal (eval différent) ; risque de fuite si mal fait.

## Decision

- **Dimension 1 → Option B** : série versionnée **G0→Vn**, harnais d'éval **gelé**.
- **Dimension 2 → Option C pour le tableau principal + Option D en diagnostic séparé.**

Les deux dimensions sont **orthogonales** : `(version de graphe) × (régime de supervision)`. La colonne « augmentée » (D) est un diagnostic, pas une entrée de leaderboard.

### Roadmap de versions (chiffrée, indicative)

| Version | Opération | Cible | Effet attendu |
|---|---|---|---|
| **G0** | brut | 87 821 art, 642k arêtes, 36 % résolus, binaire | référence |
| **V1** | retirer les 41 824 articles doublement morts du pool de candidats | catalogue → ~46 000 vivants | denoise (moins de faux positifs aléatoires) |
| **V2** | améliorer la résolution texte (bug `code_rural` 0 %, préfixes L/R/D, renumérotation) | 36 % → ? % embeddés | init plus forte, meilleure convergence |
| **V3** | filtre temporel (vieilles JP, articles abrogés) | retire la traîne historique | ⚠️ change le *scope* (voir garde-fous) |
| **V4** | pondérer les arêtes (fréquence de citation) | binaire → comptes | propagation plus fine |
| **V5+** | typage d'arêtes, résolution d'entités, dédup | — | capacité accrue |

### Protocole de l'expérience D (« voir la différence »)

Objectif : *quantifier le gain apporté par la donnée de qualité (JP+article), notamment côté JP.*

- **D-simple** (rapide) : déplacer ~50 % de la cohorte vers le train (train 1701 → ~2550, **+50 %**), évaluer sur la **moitié held-out restante**. Table à côté, mentionnée « bench réduit, indicatif ».
- **D-rigoureux** (recommandé) : **k-fold CV (5 folds)** sur la cohorte — train = pool(1701) + 4/5 cohorte, éval = 1/5 cohorte, moyenné. Utilise **toute** la bonne donnée, **zéro fuite**, faible variance. Table à côté, mentionnée « protocole CV ≠ bench gelé ».
- Dans les deux cas : **jamais évaluer une question vue au train**. Le « tableau à côté » signale un *eval différent*, pas une licence de fuite.

## Justification

- G0→Vn transforme nos défauts de données en programme de recherche mesurable : c'est la **contribution méthodologique** du Chantier 2.
- Séparer les deux dimensions évite de confondre « le graphe s'est amélioré » et « on a ajouté de la donnée ».
- L'expérience D est le **seul** moyen de tester l'hypothèse de M. Kaeppelin (« la bonne donnée JP+article améliore le tout ») et de débloquer une supervision JP, tout en restant honnête via la table séparée.

## Consequences

- **Garde-fou 1 — Harnais gelé** : même cohorte, même `metrics.py`, même split, figés dès G0. Seul le graphe change entre Vn.
- **Garde-fou 2 — Anti *scope reduction*** (surtout V3) : mesurer sur un **dénominateur GT figé** (les GT de G0) et **reporter la couverture** (% GT atteignables) à côté de chaque métrique. Une version n'est « meilleure » qu'à **couverture ≥**. Sinon retirer des GT durs flatte la Recall sans progrès réel.
- **Garde-fou 3 — Anti-fuite** (expérience D) : toute question d'éval doit être hors de son train.
- **Dépendance** : nécessite le panel `metrics.py` (Hit/MRR/NDCG) — non encore implémenté (cf. `sota-gnn-reco-2026` §4).
- **À valider avec Johnny** : (a) le principe G0→Vn ; (b) le choix D-simple vs k-fold ; (c) la définition de V1 (liste des nœuds morts à retirer).

## Sources
- [[LightGCN-2020]]
- [[sota-gnn-reco-2026]]
- [[handoff-LightGCN]]
- Mémoire projet : `lightgcn-data-reality`, `johnny-week9-decisions`
