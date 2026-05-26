---
date: 2026-05-26
type: diagnostic-branche
branche: administratif
source: 05-Technique/benchmark/etape1_embedding_pur/data/branch_diagnostics.json
encoder: BGE-M3
pool_size: 31357
threshold: 0.5
---

# Diagnostic — branche administratif (8 questions)

> **0/8 questions passent le critère dur (recall@10 core ≥ 0.5).**
> Pourquoi ? Trois causes cumulatives : **gold inatteignable** (articles hors LEGI, dont code_rural et CGFP), **gold dispersé** (3 à 7 articles core par question, dans 2–4 codes différents) et **vocabulaire compositionnel** (notions doctrinales — motivation, voie de fait, transaction administrative — qui n'apparaissent pas dans le texte des articles).

## Couverture du gold dans LEGI

Sur **23 articles "core"** attendus pour les 8 questions, **7 (30 %) sont absents du pool LEGI** :

| Code source | Articles core manquants | Détail |
|---|---|---|
| `code_general_de_la_fonction_publique` (CGFP) | **4** | L530-1, L111-1, L531-1, L533-1 (tous Q2) |
| `code_rural_et_de_la_peche_maritime` | **2** | R253-6 (Q1A), R253-5 (Q1B) |
| `code_des_relations_entre_le_public_et_l_administration` (CRPA) | **1** | L423-1 (PROC-Q2) |

Sur 26 gold "expected" : 8 manquants (mêmes codes, +1 R253-5 supplémentaire en Q1A).

**Constat-clé** : la couverture LEGI 0 % de `code_rural` n'explique qu'une **petite partie** de l'échec (2/7 = 29 % des manquants). Le **CGFP** est en réalité le code le plus impactant (4/7 = 57 % des manquants, tous concentrés sur Q2). La conclusion : **le problème n'est pas seulement code_rural**, c'est un défaut de couverture LEGI sur les codes administratifs récents (CGFP : entré en vigueur 2022, ré-incorporation incomplète dans le pool ?).

Par ailleurs, **2 questions sur 8 ont un gold core vide** (Q1C, PROC-Q3) : elles s'évaluent uniquement contre la doctrine / JP et ne peuvent par construction pas être satisfaites en retrieval pur d'articles.

## Vue d'ensemble

| Question | Thème | gold_core | absents pool | K\*_core (o / f) | r@10 core (o / f) | top1 gold rank (o / f) | pattern dominant |
|---|---|---|---|---|---|---|---|
| ADMIN-Q1A | AAU, motivation | 4 | 1 (rural) | None / 1000 | 0 / 0 | 1939 / 250 | gold dispersé + vocab compo |
| ADMIN-Q1B | police générale vs spéciale | 2 | 1 (rural) | None / 1000 | 0 / 0 | 4668 / 820 | article-principe + rural absent |
| ADMIN-Q1C | délégation police à privé | 0 | — | None / None | — | — | gold core vide (constitutionnel/JP) |
| ADMIN-Q2 | discipline fonctionnaire | 6 | 4 (CGFP) | None / None | 0 / 0 | 3705 / 421 | couverture CGFP cassée |
| ADMIN-Q3 | resp. sans faute, attroupements | 2 | 0 | None / 200 | 0 / 0 | 1027 / 185 | vocab compo (responsabilité) |
| PROC-Q1 | délais, injonction | 7 | 0 | None / 200 | 0 / 0 | 358 / 66 | gold trop dispersé (7 articles) |
| PROC-Q2 | contrat admin, transaction | 2 | 1 (CRPA L423-1) | 100 / 30 | 0 / 0 | 79 / 23 | proche du seuil, CRPA absent |
| PROC-Q3 | voie de fait, emprise | 0 | — | None / None | — | — | gold core vide (compétence/JP) |

Légende : `o` = open (31 357 chunks), `f` = filtered (pool restreint par codes admin, ~4 253 chunks).

## Détails par question

### CNB-ADMIN-2025-Q1A — annulation arrêté ministériel (motivation)
- **Gold core (4)** : R253-6 rural, L200-1 / L211-3 / L211-5 CRPA. **R253-6 absent du pool.**
- **Ranks open** : L211-3 = 1939, L211-5 = 6202, L200-1 = 17670. **Aucun gold dans le top-1000.**
- **Top-5 filtré** : CGCT L2213-5, CSI L612-16, CJA L522-3, CJA R811-15, CRPA L241-2 — tous "périphériques admin" mais hors gold.
- **Diagnostic** : (i) R253-6 inatteignable ; (ii) la question parle d'« arrêté », « annulation », « article 53 règlement 1107/2009 », « motivation » — l'encodeur fait remonter des articles avec « arrêté » + « interdire » (CGCT police municipale) plutôt que l'article-principe L211-3 CRPA (obligation de motivation des décisions individuelles défavorables) qui ne contient pas explicitement le mot « motivation » dans son intitulé sémantique. **Article-principe abstrait + vocabulaire compositionnel.**
- **Fix** : indexer la table des matières CRPA (« motivation des décisions administratives ») comme contexte additionnel ; ajouter R253-6 au pool.

### CNB-ADMIN-2025-Q1B — police municipale vs ministérielle
- **Gold core (2)** : CGCT L2212-2, code_rural R253-5. **R253-5 absent.**
- **Ranks** : L2212-2 = 4668 open / 820 filtré. Le top-5 filtré contient L2213-4, L2213-2, L2213-3, L2213-5 (police municipale spéciale) — **proches sémantiquement mais à côté du gold L2212-2 (police générale).**
- **Diagnostic** : code_rural absent + **code-confusion fine** : l'encodeur préfère L2213-x (police municipale spécialisée) à L2212-2 (police générale) car la question évoque des matières dangereuses, l'environnement, etc.
- **Fix** : pool rural ; ranker conscient de la hiérarchie « générale > spéciale » ou MMR pour diversifier sur les sous-articles.

### CNB-ADMIN-2025-Q1C — délégation police admin à privé
- **Gold core vide** : la grille CNB attendue est constitutionnelle (Conseil constit., identité constitutionnelle) + JP TC.
- **Diagnostic** : **inévaluable par retrieval d'articles**. Top-5 filtré (L2333-74 CGCT, R232-15 CSI, L611-1 CSI) est plausible mais sans gold.
- **Fix** : exclure cette question du calcul du recall, ou annoter manuellement une gold "JP/constit".

### CNB-ADMIN-2025-Q2 — discipline fonctionnaire, liberté d'expression
- **Gold core (6)** : tous CGFP — L3, L121-1, L530-1, L111-1, L531-1, L533-1. **4 absents du pool** (L530-1, L111-1, L531-1, L533-1).
- **Ranks** : L3 = 8325 / 863, L121-1 = 3705 / 421. Aucun dans top-100.
- **Top-5 filtré** : CGCT L2213-2, CSI R612-4, CGCT L2123-34, route R412-11, CJA L521-2 — **totalement à côté**.
- **Diagnostic** : **couverture LEGI du CGFP gravement incomplète** (2 articles sur 6 indexés) ; en plus les 2 articles présents ne remontent pas car la question évoque « réseau social X », « khmers verts », « suspension » — vocabulaire qui n'apparaît pas dans les articles CGFP très abstraits (« exerce ses fonctions avec dignité, impartialité, intégrité… »). **Couverture cassée + article-principe abstrait.**
- **Fix prioritaire** : **vérifier l'ingestion CGFP** (probablement le code le plus impactant à corriger sur cette branche).

### CNB-ADMIN-2025-Q3 — responsabilité sans faute, attroupements
- **Gold core (2)** : CSI L211-10, route L412-1. **Tous résolus**.
- **Ranks open** : 2231 / 1027 ; filtré : 372 / 185. **K\*_core = 200 (filtré).**
- **Top-5 filtré** : route L325-1, CJA R811-1-1, civil 1785, civil 645, CGCT L2212-2-2.
- **Diagnostic** : couverture OK mais **vocabulaire compositionnel** — la question parle de « barrages routiers », « perte de recettes », « régime de responsabilité » ; L211-10 CSI est l'article-principe sur la responsabilité de l'État pour attroupements — mais le texte de l'article ne reprend pas les termes de la question. L'encodeur remonte des dispositions police de la route ou civiles.
- **Fix** : enrichir le contexte (titre de section « Responsabilité en cas d'attroupements et de rassemblements ») au moment de l'indexation.

### CNB-PROC-ADMIN-2025-Q1 — délais, injonction (gros gold dispersé)
- **Gold core (7)** : 4× CJA (R421-1/-2/-5, R431-2, L911-1) + 2× CRPA (L411-2, L411-7). **Tous résolus.**
- **Ranks filtré** : R421-1 = 66, R421-2 = 73, R421-5 = 103, L411-2 = 162, L411-7 = 184, R431-2 = 892, L911-1 = 1100. **K\*_core = 200** (filtré), r@100 = 0.286.
- **Top-5 filtré** : tous code de la route (R222-3, R224-4, etc.) — la question parle de « permis de conduire au Maroc, échange ». **L'encodeur s'accroche au lexique permis/conduire** au détriment du squelette procédural.
- **Diagnostic** : **lexical OK mais distracteurs forts** (code route) + gold trop dispersé (7 articles → r@10 ≥ 0.5 demande 4 articles dans le top-10, mathématiquement très difficile sans reranking).
- **Fix** : intent classifier en amont (« question procédurale → écarter code route ») ; ou poser le retrieval comme cascade « code procédure d'abord, fond ensuite ».

### CNB-PROC-ADMIN-2025-Q2 — contrat admin, transaction
- **Gold core (2)** : CRPA L423-1, commande publique L6. **L423-1 absent du pool.**
- **Ranks filtré** : L6 = 23. **K\*_core = 30 (filtré).** La question est à un seul article près du seuil mais ne passe pas car L423-1 inatteignable.
- **Diagnostic** : **plus proche succès de la branche** — si CRPA L423-1 était indexé et remontait dans le top-10, recall@10 serait 0.5. La question évoque explicitement « transaction » qui est dans le titre de la section L423.
- **Fix** : **vérifier l'ingestion CRPA L4xx** (problème ciblé : la section transactions du CRPA semble lacunaire).

### CNB-PROC-ADMIN-2025-Q3 — voie de fait, emprise irrégulière
- **Gold core vide.** Gold expert = L521-1/L521-2 CJA (référé), mais le cœur est JP (Bergoend 2013, TC 12 juin 2023).
- **Diagnostic** : **inévaluable par retrieval d'articles** (question 100 % JP/répartition de compétence).
- **Fix** : exclure du recall ou annoter une gold de substitution.

## Synthèse

- **Pattern dominant** : **couverture LEGI lacunaire (CGFP + code_rural + un article CRPA) + gold dispersé sur ≥ 4 codes + articles-principes abstraits**. Le retrieval pur BGE-M3 ne peut pas réussir sur ce type de question CNB.
- **Part d'échec due à code_rural absent** : 2 questions sur 8 sont contaminées (Q1A, Q1B) mais aucune n'aurait passé le critère dur même avec rural indexé, car les autres articles gold sont déjà au-delà du rang 200 en filtré. **Code_rural = facteur aggravant, pas la cause principale.**
- **Part d'échec due au CGFP** : 1 question (Q2) lourdement impactée — 4/6 gold core inatteignables. **CGFP = priorité d'ingestion**.
- **Part d'échec due à l'encodeur (questions où la couverture est OK)** : Q3, PROC-Q1, et au sens large Q1A (3/4 articles présents). Sur ces 3 questions, les gold sont dans le pool mais hors top-100 ; un reranker ou un retrieval hybride dense+lexical aurait des chances de remonter le top-10.
- **Questions inévaluables** : Q1C et PROC-Q3 ont un gold core vide (= 25 % de la branche). Compter ces deux questions comme « échecs » est biaisé.

### Recommandations

1. **Prioriser l'ingestion CGFP et CRPA L4xx** dans le pool (gain potentiel immédiat : 4 questions sur 8).
2. **Exclure Q1C et PROC-Q3** des métriques globales ou leur attribuer une gold construite (article-principe + JP de référence).
3. **Côté encodeur** : sur PROC-Q1 et Q3, tester (i) un retrieval lexical BM25 en complément (les numéros d'articles cités dans la question — L. 112-3, R. 112-5, art. 53 reg. 1107/2009 — ne sont pas exploités), (ii) une étape de reranking cross-encoder.
4. **Côté évaluation** : pour les questions à 7+ articles gold core, le critère « recall@10 ≥ 0.5 » est très exigeant ; envisager r@20 ou r@50 comme critère secondaire pour ne pas masquer un retrieval « OK mais dispersé ».

**Bilan** : sur 8 questions, ~3 échecs strictement attribuables à **encodage** (Q3, PROC-Q1, et partiellement Q1A pour les CRPA dispersés), ~3 échecs dominés par **couverture LEGI** (Q1B, Q2, PROC-Q2), et 2 questions **inévaluables** par retrieval d'articles (Q1C, PROC-Q3).
