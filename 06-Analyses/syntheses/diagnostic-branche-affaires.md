---
date: 2026-05-26
type: diagnostic-branche
branche: affaires
---

# Diagnostic — branche affaires (4 questions)

Source : `05-Technique/benchmark/etape1_embedding_pur/data/branch_diagnostics.json` (BGE-M3, pool 31 357, branche affaires = code_civil + code_de_commerce + code_monetaire_et_financier). Critère dur : tous les gold `obligatoires` doivent être dans le top-K.

## Vue d'ensemble

| Question | spécialisation | gold_core | gold_oblig | K\*_core (o/f) | R@10 core (o/f) | top1 gold rank (o/f) | pattern |
|---|---|---|---|---|---|---|---|
| Q1 | sociétés — pourparlers, formation | 2 | 2 | 1 / 1 | 0.50 / 0.50 | 1 / 1 | Article-principe abstrait (1112 absent) |
| Q2 | SAS — clause d'agrément, transfert | 4 | 4 | 100 / 100 | 0.00 / 0.00 | 69 / 50 | Vocabulaire absent + question multi-articles |
| Q3 | entreprises en difficulté — liquidation, bail | 4 (5 oblig) | 5 | 50 / 30 | 0.25 / 0.25 | 3 / 3 | Trop d'articles gold + Article-principe (L622-6) |
| Q4 | bail commercial — clause résolutoire | 3 | 3 | ∞ / ∞ (R@1000=0.33) | 0.00 / 0.00 | 263 / 94 | Code-confusion massive + Vocabulaire absent |

## Détails par question

### CNB-AFFAIRES-2025-Q1 — sociétés, pourparlers rompus

**Question (extrait)** : Louis est écarté avant signature des statuts, menace une action en réparation. Quels risques pour les associés ?
**Gold core** : `code_civil:1832` (rang 1), `code_civil:1112` (rang 6363 open / 1891 filtered).
**Top-5 open** : 1832 (OBLIG), 1833, R210-5 (C. com.), L322-1 (assurances), R223-32 (C. com.).
**Top-10 distractions** : 1833, R210-5, L223-6, L225-206, 1844-7, L223-1 — variantes "constitution de société". L322-1 (assurances) hors branche en open mais filtre branche le retire.

**Diagnostic** : succès sur l'article-définition (1832 = "la société est instituée…"), échec total sur 1112 (négociations précontractuelles). La question ne contient aucun mot du vocabulaire de 1112 ("pourparlers", "bonne foi", "rompre les négociations") — elle dit "écarté juste avant signature". BGE-M3 ne fait pas le pont sémantique question-narrative → article-principe abstrait. Pattern : **Article-principe abstrait + Vocabulaire absent**.
**Fix** : réécriture LLM de la question pour expliciter "responsabilité pour rupture des pourparlers" ; à défaut multi-vecteur (HyDE) ou recall@50 ne suffit pas (rang 1891 même filtré).

### CNB-AFFAIRES-2025-Q2 — SAS, clause d'agrément, transfert

**Question (extrait)** : Cession d'actions au petit-fils, clause statutaire d'agrément avec dispense ascendants/descendants. Convocation refusée.
**Gold core** : L227-14 (rang 69/50), L227-15 (97/75), L228-1 (903/591), L211-17 CMF (2760/1356). **Aucun gold dans le top-50** (open). R@10 = 0.
**Top-10 open** : 1861 (parts sociales), L227-16 (exclusion d'associé), L223-14 (parts SARL), R223-12, R212-3 CCH, 1868, L228-6-3, L225-197-1, L227-18, R221-9. En filtered, L228-23 (EXPECTED) apparaît rang 10.

**Diagnostic** : énorme **code-confusion** entre régimes d'agrément (SARL parts sociales L223-14 / SAS L227-14-16 / société civile 1861). Le modèle retourne le mauvais régime alors que la question dit "SAS" et "actions". L228-1 (titres) et L211-17 CMF (transfert de propriété par inscription) sont des articles techniques sans recouvrement lexical avec la narrative — pattern **Vocabulaire absent** (la question ne dit pas "inscription en compte"). Le piège annoté insiste précisément sur L228-1/L211-17 ("3 pts sur 5").
**Fix** : (a) filtre dur "SAS → exclure articles L223-* (SARL) et 18xx (sociétés civiles)" infaisable sans typage fin ; (b) réécriture LLM injectant "inscription en compte, transfert de propriété des titres" ; (c) recall@100 récupère 2/4 — toujours insuffisant. Cas qui demande hybridation BM25 (L227-14/15 contiennent "agrément" — devraient remonter en lexical pur).

### CNB-AFFAIRES-2025-Q3 — liquidation judiciaire, bail, déclaration de créance

**Question (extrait)** : SAS en liquidation ; bailleur invoque résiliation, banque clôt le compte, dirigeant hésite à déclarer une dette contestée.
**Gold core** : L622-14 (45/29), L641-12 (3/3), L641-11-1 (90/56), L622-6 (1872/825). Gold oblig +L622-13 (660/345).
**Top-10 open** : L511-17 CMF, L641-3, **L641-12 (rang 3, OBLIG)**, L643-1 (EXPERT), L645-9, L326-1 assurances, L742-20 conso, L724-3 conso, L640-1, L326-13 assurances. Filtered nettoie les codes hors-branche.

**Diagnostic** : seul L641-12 (résiliation du bail en liquidation) sort fort — recouvrement lexical direct "liquidation/bail". L622-14 (continuation contrats) et L641-11-1 (contrats en cours en liquidation) sont enterrés (rangs 45-90) — partagent peu de mots avec la narrative. L622-6 à 1872 = **article-principe** (inventaire) sans aucun mot-clé dans la question. **Trop d'articles gold (5)** sur une question multi-thématique (A bail + B compte + C créance) → le modèle ne peut pas tout couvrir avec une seule requête. Code-confusion modérée en open (codes assurances et consommation pour "liquidation" non-procédure-collective) que le filtre branche corrige bien.
**Fix** : **décomposition de la question en 3 sous-requêtes** (A bail / B compte bancaire / C déclaration de créance) — c'est le levier principal. Recall@50 filtered = 0.5 → pas suffisant ; recall@100 = 0.75 ; L622-6 nécessite réécriture explicite.

### CNB-AFFAIRES-2025-Q4 — bail commercial, clause résolutoire, exception d'inexécution

**Question (extrait)** : Bail commercial, tempête, locataire suspend les loyers, commandement de payer, clause résolutoire.
**Gold core** : `code_civil:1225` (263/94), `code_civil:1219` (11569/2497), `code_civil:1732` (11134/2407). **Aucun gold dans le top-200** (filtered). R@10 = 0, R@500 = 0.33.
**Top-10 open** : L714-1 conso (impayé locatif), 31 CGI (charges propriété), L312-40 conso, L4311-5 travail, L412-3 PCE, 39 CGI, L251-7 CCH, L145-39 C. com., L57 postes, L145-47 C. com. → **code-confusion catastrophique**. Top-10 filtered (codes affaires) : L145-39, L145-47, L145-58, L145-7, L642-17, L145-23-1, **L145-41 (rang 7, EXPECTED)**, L145-9, L470-1, L145-14 — uniquement des articles spécifiques bail commercial L145-*, jamais les articles de droit commun visés par le gold.

**Diagnostic** : le piège annoté est exactement ce qui se produit — le modèle plonge dans L145-* (régime spécial bail commercial) au lieu des articles de droit commun (1225 clause résolutoire / 1219 exception d'inexécution / 1732 dégradations du preneur). **Code-confusion + Vocabulaire absent** : la question dit "clause résolutoire" → BGE va chercher L145-41 (clause résolutoire de bail) au lieu de 1225 (régime général). 1219 et 1732 n'ont aucun ancrage lexical dans la question ("suspendre le loyer" ≠ "exception d'inexécution"). C'est l'**échec le plus profond** des 4 (rangs au-delà de 2400 même filtrés).
**Fix** : réécriture LLM impérative pour expliciter le vocabulaire civiliste ("exception d'inexécution", "résolution unilatérale", "restitution des locaux dégradés"). Filtre code seul ne suffit pas (1225/1219/1732 sont bien dans `code_civil` filtré). Probablement nécessiter un **reranker cross-encoder** entraîné sur des paires narrative→articles, ou décomposition LLM en sous-questions juridiquement qualifiées.

## Synthèse

- **Pattern dominant** : **Vocabulaire absent + code-confusion** — la narrative CRFPA reste factuelle/civile, tandis que les articles gold supposent un vocabulaire juridique de qualification (pourparlers, exception d'inexécution, inscription en compte, contrats en cours) que la question ne contient jamais. BGE-M3 ne fait pas le pont narrative→qualification.
- **Questions sauvables par fix simple** : Q1 (réécriture LLM injectant "rupture pourparlers / responsabilité précontractuelle") et Q3 (décomposition en 3 sous-questions). Q2 nécessite hybridation BM25 ou rerank. Q4 = cas le plus dur, nécessite réécriture profonde + rerank.
- **Recommandation** : (1) prototyper une **étape de réécriture LLM** (question → mots-clés juridiques qualifiés + décomposition en sous-requêtes) avant embedding ; (2) tester **BM25 hybride** (les pair_keys L227-14/15/L228-1 contiennent les mots-clés exacts du gold) ; (3) garder en tête que **recall@50 filtered ne dépasse jamais 0.5** sur 3/4 questions affaires — l'embedding pur a un plafond structurel sur cette branche.
