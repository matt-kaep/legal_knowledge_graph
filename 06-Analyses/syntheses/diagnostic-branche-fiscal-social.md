---
date: 2026-05-26
type: diagnostic-branche
branches: [fiscal, social]
source: 05-Technique/benchmark/etape1_embedding_pur/data/branch_diagnostics.json
modele: BGE-M3 (etape1 embedding pur)
seuil: 0.5
---

# Diagnostic — branches fiscal (3 q) et social (3 q)

Lecture : `K*_core` = plus petit k ou recall_core = 1. `r10_core` = recall@10 sur le set "core". `top1_rank` = rang du premier gold core. `o/f` = open / filtered (filtre branche).

Pool open = 31 357 articles. Pool filtré fiscal = 3 224. Pool filtré social = 7 293.

## Vue d'ensemble

| Question | branche | gold_core | K*_core (o/f) | r10_core (o/f) | top1_rank (o/f) | pattern |
|---|---|---|---|---|---|---|
| FISCAL-Q1 | fiscal | 7 (6 résolus) | – / – (jamais 1.0) | 0.14 / 0.14 | 2 / 2 (CGI:271) | Trop d'articles gold + 1 hors-LEGI (293A) + article-principe abstrait (259) |
| FISCAL-Q2 | fiscal | 11 | – / 500 | 0.27 / 0.27 | 1 / 1 (CGI:32) | Question multi-personae très technique, gold massif (11 art) |
| FISCAL-Q3 | fiscal | 1 (CGI:93) | 1000 / 200 | 0.0 / 0.0 | 506 / 109 | Article-principe abstrait : "BNC" générique ≠ "acte anormal de gestion" |
| SOCIAL-Q1 | social | 2 | – / – | 0.0 / 0.0 | 276 / 234 | Vocabulaire absent : 3 sous-cas, articles-principes (L1132-1 discrim) |
| SOCIAL-Q2 | social | 1 (CSS:L411-2) | 1 / 1 | 1.0 / 1.0 | 1 / 1 | Lexical OK — match parfait "accident de trajet" |
| SOCIAL-Q3 | social | 4 | – / – | 0.0 / 0.0 | 25 / 25 (L2314-10) | Question 3-en-1, articles techniques CSE numérotés |

"–" = recall_core ne saturer jamais à 1.0 dans la fenêtre [1..1000].

## Cas particulier SOCIAL-Q2 — pourquoi ça marche ?

**Seul succès des 6 (rang 1 open ET filtered, K\*=1).** Trois propriétés convergent :

1. **Mono-gold core** : un seul article (`code_de_la_securite_sociale:L411-2`). Pas de problème combinatoire.
2. **Vocabulaire de la question = vocabulaire de l'article**. La question contient littéralement "accident de trajet" et "legislation sur les accidents de trajet". L'article L.411-2 du CSS est l'article qui *définit* le trajet protégé et contient la même expression. Le label `code_de_la_securite_sociale` + le terme "trajet" sont rares dans le corpus : peu de bruit.
3. **Question courte et factuelle** (8 lignes), un seul scénario (M. Stefan déneige sa voiture, chute, demande la qualification AT/trajet). Pas de sous-questions, pas de personae multiples.

À comparer aux 5 autres :
- FISCAL-Q1 : 6 opérations TVA indépendantes (LASM, indemnité, prestation, importation, livraison intra-FR, prestation hors UE) → la question agrège 6 sujets juridiques, le vecteur sémantique est "moyen".
- FISCAL-Q2 : 2 personae × 5 catégories de revenus.
- SOCIAL-Q1 : 3 sous-cas (refus de roulement, propos racistes, prime anti-grève).
- SOCIAL-Q3 : 3 sous-cas (ASC du CSE, vacance de siège, transparence syndicale).
- FISCAL-Q3 : doctrine (acte anormal de gestion sur BNC) non écrite dans CGI:93.

**Le facteur discriminant n'est pas la branche** : c'est (1) le nombre de sujets juridiques dans la question, (2) le recouvrement lexical question/article, (3) le caractère principe-vs-spécifique de l'article gold.

## Détails par question

### CNB-FISCAL-2025-Q1 — TVA, 6 opérations
- **K\*_core** : jamais 1.0 ; recall@1000 = 0.71 (open et filtered). `code_general_des_impots:293A` jamais résolu (gold_missing → article inexistant en LEGI ou normalisation pair_key cassée).
- **Top1 core** : CGI:271 rang 2 (open & filtered).
- **Gold hors top-50 (open)** : 256 (#64), 269 (#227), 259 (#13316), 291 (#159), 287 absent.
- **Bruit top-10** : CGI:1695 (#1), CGI:39, CGI:289-0, CGI:1693, CGI:261, CGI:257ter → articles TVA *connexes* mais hors-gold ; le modèle "voit" TVA mais n'arrive pas sur les articles-cadres (256/259).
- **Pattern** : trop d'articles gold (7) + 1 hors-LEGI (293A) + article-principe abstrait (259 = règles de territorialité PS, rang 13316 hallucinant).
- **Fix** : (a) résoudre 293A dans LEGI (vérifier numérotation), (b) accepter une marge tolérante sur "art. 256 ↔ 256bis/ter" pour évaluation, (c) accepter recall@100 = 0.43 comme baseline réaliste sur ce type de question multi-opérations.

### CNB-FISCAL-2025-Q2 — IR, foyer fiscal Lemoine
- **K\*_core** : 500 open / 30 filtered. Recall_core@100 = 0.55 open / 0.64 filtered, @500 filtered = 1.0.
- **Top1 core** : CGI:32 rang 1 (revenus fonciers micro-foncier : la question contient "loyers", "18 000 EUR", "appartement" → match fort).
- **Gold hors top-50 (open)** : CGI:79 (#691), CGI:14 (#1155), CGI:92 (#1235), CGI:200 (#247), 4A/4B/102/196B absents.
- **Bruit top-10 (open)** : CGI:170bis (déclaration), 224, A444-83 (commerce), 158, 168, D553-1 (CSS prestations familiales) → bruit massif sur articles déclaratifs.
- **Pattern** : question multi-personae, gold massif (11 core), articles-principes de qualification (4A = personne imposable, 4B = domicile fiscal) jamais récupérés.
- **Fix** : filtrage branche déjà aide nettement (K* divisé par 17). Le vrai gain viendrait d'un découpage de la question en sous-requêtes "revenus salariés / revenus fonciers / BNC / pension alimentaire / don".

### CNB-FISCAL-2025-Q3 — Acte anormal de gestion sur BNC
- **K\*_core** : 1000 open / 200 filtered. **Recall@10 = 0** dans les deux.
- **Top1 core** : CGI:93 rang 506 open, 109 filtered.
- **Bruit top-10** : CGI:98, R314-146 (action sociale), R444-14, CGI:1727 (intérêts de retard), 1960, R743-147, CGI:95, **livre_des_procedures_fiscales:L247** et **L203** → le modèle s'oriente vers "contrôle fiscal / redressement" et non BNC.
- **Pattern** : article-principe abstrait. **CGI:93 ne mentionne ni "acte anormal de gestion" ni "tarif réduit entre confrères"**. La doctrine (CE 23 déc. 2013) n'est nulle part dans le texte. Question piège : le candidat doit savoir que la théorie *ne s'applique pas* aux BNC, donc le seul article pertinent est l'article général sur le BN-bénéfice.
- **Fix** : pour ce type de question doctrinale, l'embedding pur est structurellement insuffisant. Nécessite (a) graphe JP→articles, ou (b) un retrieval sur la doctrine (BOFIP).

### CNB-SOCIAL-2025-Q1 — TRANSFRAIS individuel (3 sous-cas)
- **K\*_core** : jamais 1.0 dans [1..500] open ; @1000 filtered = 1.0.
- **Top1 core** : L1132-1 rang 276 open / 234 filtered.
- **Gold hors top-50** : L1132-1 (#276), L1134-1 (#1261), L1121-1 (#25710 !), L1235-3-1 (#354), L4121-1 (#4364).
- **Bruit top-10** : L3133-10 (jours fériés), L3122-12 (travail de nuit — *intuitivement plausible* car la question parle de "agent des services incendie en poste de nuit"), R3261-8, L5222-2, R433-11 (CSS AT), L1235-13.
- **Pattern** : vocabulaire absent. La question est très narrative (3 affaires "Garcia / Kaddouche / prime"), et les articles gold sont des **articles-principes du livre I** (non-discrimination, libertés). Le terme "discrimination" n'est utilisé qu'implicitement ("propos racistes"). L1121-1 (libertés individuelles) à rang **25 710** : l'article est trop abstrait pour matcher un cas concret.
- **Fix** : reformulation type "discrimination raciste, libertés individuelles" + découpage en 3 sous-questions ; ou utiliser un classifieur thématique amont.

### CNB-SOCIAL-2025-Q2 — Accident de trajet (cf. analyse ci-dessus)
- **K\*_core = 1** (open & filtered). Recall_core@1 = 1.0. **Seul cas qui passe.**
- Voir section dédiée.

### CNB-SOCIAL-2025-Q3 — TRANSFRAIS collectif (3 sous-cas)
- **K\*_core** : jamais 1.0 ; @1000 = 0.75.
- **Top1 core** : L2314-10 rang 31 open / 25 filtered (meilleur que SOCIAL-Q1 grâce au terme "élection").
- **Gold hors top-50** : L2132-3 (#860), L2314-29 (#225), L2314-33 (#2065), R2314-19 (#4644).
- **Bruit top-10 (filtered)** : L1457-1 (prud'hommes), L2232-3 (négo collective), L2122-10-6, L2121-2 (rep. syndicale), L2314-15 (élections), L2141-5, L1144-2, L2314-32, R731-8, L2232-24.
- **Pattern** : question 3-en-1 sur articles techniques très numérotés (L2314-xx = élections CSE). Le bruit top-10 contient *beaucoup* de L23xx-yy plausibles, mais pas les bons. Surreprésentation, vacance de siège, transparence financière : 3 sujets distincts.
- **Fix** : découpage en sous-questions ; ou hybride lexical (BM25) qui matche le numéro d'article si cité dans la question.

## Synthèse comparée

**Fiscal — pattern dominant** : (a) gold massif (7-11 articles core) sur questions cas-pratique TVA/IR multi-opérations ; (b) articles-principes très numérotés (CGI:4A, 4B, 256, 259) au texte abstrait que la question concrète ne touche pas ; (c) 1 article gold non-résolu (293A) traduisant un problème de normalisation pair_key. Le **CGI a ~4 000 articles** mais le pool filtré fiscal est à 3 224 → le filtre branche divise K* par 2 à 4 sans atteindre le top-10. Constat : sur 3 questions, 0 succès @10.

**Social — pattern dominant** : (a) articles-principes du livre I (L1121-1, L1132-1) hors top-200 car vocabulaire question (narratif) ≠ vocabulaire article (abstrait) ; (b) succès quand 1 seul gold + vocabulaire exact (Q2). Le contraste Q2 vs Q1/Q3 est net : 1 article spécifique vs 4-5 articles abstraits.

**Recommandations** :
1. **Découpage des questions multi-sujets en sous-requêtes** (gain attendu surtout sur FISCAL-Q1/Q2, SOCIAL-Q1/Q3).
2. **Hybride lexical (BM25) + dense** : capter les numéros d'article ou termes techniques rares (CGI:293A, "accident de trajet", "L2314-33").
3. **Tolérer recall@100 plutôt que @10** comme métrique cible sur les questions à gold > 5 articles — c'est une limite structurelle de l'embedding sur cas-pratiques agrégatifs.
4. **Cas FISCAL-Q3 (doctrine)** : signaler que l'évaluation embedding-pur n'a aucun sens sur les questions "anti-piège" où l'article gold ne contient pas le concept testé (acte anormal de gestion ∉ CGI:93).
5. **Corriger la normalisation pair_key pour CGI:293A et CGI:287** (gold non résolus).
