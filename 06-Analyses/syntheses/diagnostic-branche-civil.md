---
date: 2026-05-26
type: diagnostic-branche
branche: civil
tags: [synthese, etape1, diagnostic, civil, retrieval]
---

# Diagnostic — branche civil (9 questions)

Données : `etape1_embedding_pur/data/branch_diagnostics.json` — BGE-M3, pool branche civil = 4430 articles.

## Vue d'ensemble

| Question | spec | gold_core | gold_oblig | K*_core (o/f) | K*_oblig (o/f) | r@10 core (o/f) | top1 gold core rank (open) | pattern |
|---|---|---|---|---|---|---|---|---|
| CIVIL-Q1 | régimes matrim. (communauté, droits sociaux) | 6 | 8 | 500 / 100 | 500 / 100 | 0.00 / 0.17 | 33 (`1424`) | Trop d'articles gold + vocab abstrait |
| CIVIL-Q2 | mandat (dépassement, apparent) | 7 | 7 | — / — | — / — | 0.00 / 0.00 | 277 (`1999`) | Vocabulaire absent |
| CIVIL-Q3 | vices cachés (vente, délai biennal) | 4 | 4 | 200 / 30 | 200 / 30 | 0.25 / 0.25 | 5 (`1644`) | Lexical partiellement OK + article hors thème (`2232`) |
| OBLIG-Q1 | vices du consentement (dol, devoir d'info) | 8 | 11 | — / — | — / — | 0.00 / 0.00 | 1087 (`1112-1`) | Vocabulaire absent (cas factuel pur) |
| OBLIG-Q2 | sanctions inexécution / resp. contractuelle | 3 | 4 | — / — | — / — | 0.00 / 0.00 | 339 (`1217`) | Vocabulaire absent |
| OBLIG-Q3 | clauses abusives / prescription B2B | 3 | 4 | — / 500 | — / 500 | 0.00 / 0.33 | 19 (`2254`) | Article-principe abstrait (1119, 1170) |
| PROCCIV-Q1 | MARD / ARA et conciliation | 7 | 7 | — / — | — / — | 0.00 / 0.00 | 133 (`1528`) | Article hors LEGI (4/7 oblig manquants : ARA décret 2025) |
| PROCCIV-Q2 | appel : preuves nouvelles, loyauté | 3 | 2 | — / 500 | 1 / 1 | 0.33 / 0.33 | 1 (`563`) | Lexical OK (oblig) ; core plafonné (recall@10 = 1/3 max) |
| PROCCIV-Q3 | interruption d'instance, curatelle | 6 | 6 | 200 / 100 | 200 / 100 | 0.17 / 0.17 | 4 (`530`) | Lexical OK partiel — `371` (exception) absent |

Format : `o/f` = open / filtered. `—` = K* > 1000 (jamais atteint). `r@10` plafond mathématique = min(1, 10/|gold|).

## Question par question

### CNB-CIVIL-2025-Q1 — régimes matrimoniaux

**Q (extrait)** : Raphael et Linda mariés sans contrat ; parts SARL, actions SA, droit de présentation libéral ; effets du divorce sur droits sociaux.

**Top-5 (open)** : `code_civil:1861` (cession parts), `code_de_commerce:L223-13/14` (SARL), `cch:L212-4`, `code_civil:1832-2` **← EXPECTED rang 6**.
**Top-10 (filtered)** : capture `code_civil:1832-2` (rang 2), `code_civil:1424` **← OBLIG rang 6** ; rate `1401, 1421, 262-1, 1442, 1427`.

**Diagnostic** : la question contient des mots-clés très saillants ("parts SARL", "actions", "associé") qui font remonter le droit des sociétés (livre III) au détriment des régimes matrimoniaux (livre III bis). `1401` (communauté d'acquêts) est rang 3415 open : article-principe trop générique. Filtered récupère partiellement (K*_core=100) mais r@10 plafonné à 1.67 = 1 gold / 6. Pattern : **trop d'articles gold + code-confusion droit des sociétés vs régimes**.

**Fix** : (a) recall@50 filtered captures 2/6 core ; (b) réécriture LLM ajoutant "régime de communauté légale", "biens communs" devrait débloquer `1401, 1421` ; (c) MMR pour casser la grappe "droit des sociétés".

### CNB-CIVIL-2025-Q2 — mandat

**Q** : Octave mandate Vincent (ami non-pro) pour vendre un appartement à 200 k€ min ; Vincent accepte 170 k€ ; opposabilité au tiers + rémunération.

**Top-5 (open et filtered)** : `code_civil:1674` (lésion vente immeuble), `1592` (prix), `1621`, `code_de_commerce:L134-x` (agent commercial). **AUCUN article du mandat (1984+) en top-50**. Le premier gold est `1999` rang 277 (open), `63` (filtered).

**Diagnostic** : échec massif. La question décrit le scénario factuel ("vente immobilière à un prix inférieur") ; BGE-M3 récupère du droit spécial de la vente (1674 lésion) et du droit de l'agent commercial (L134) — il n'identifie jamais qu'il s'agit d'un **mandat civil non-professionnel**. Les articles 1984-1999 (du mandat) utilisent un vocabulaire abstrait ("pouvoir", "mandataire") que la question n'invoque pas explicitement. Pattern : **vocabulaire absent (compositionnel pur, A)**.

**Fix** : **réécriture LLM obligatoire** — ajouter "mandat", "mandataire", "représentation". Aucun fix simple ne sauve cette question.

### CNB-CIVIL-2025-Q3 — vices cachés véhicule occasion

**Q** : Octave achète véhicule d'occasion ; vices ; action récursoire ; délai biennal.

**Top-10 (open)** : `code_de_la_route:L327-3, L322-2`, `code_consommation:L241-5`, `code_civil:570`, **`code_civil:1644` ← OBLIG rang 5**. Filtered : `1644` rang 2, `1645, 1646` (EXPECTED) rangs 8-9.

**Diagnostic** : succès partiel — `1644` (vice caché, choix acheteur) trouvé très haut ; mais `1641` (définition) rang 1159 open / 164 filtered, et `2232` (délai-butoir 20 ans) rang 11575 open. Le top-10 contient du Code de la route (factuellement pertinent : véhicule) qui dilue. Pattern : **lexical partiellement OK + code-confusion (Code de la route)**.

**Fix** : filtered + recall@30 → r=0.5 oblig. Acceptable. MMR ou exclusion code_de_la_route dans le filtre code.

### CNB-OBLIG-2025-Q1 — dol et devoir d'information

**Q** : Mme des Pres acquiert propriété ; festival bruyant non révélé ; dol/réticence dolosive.

**Top-10 (open)** : entièrement hors-sujet — `cpc_exec:L221-3` (vente forcée), `env:L514-20` (ICPE), `minier:L154-2` (mine), `commerce:L145-46-1`. Filtered : `civil:1674, 639, 2346` — toujours rien sur vices du consentement. Premier gold : `1112-1` rang 1087 open / 1559 filtered.

**Diagnostic** : échec catastrophique (r@500 = 0). La question est un cas factuel sans aucun terme juridique de référence ("dol", "erreur", "consentement", "vice"). BGE-M3 latch sur "acquérir une propriété" → tout le droit de la vente immobilière et des ICPE. **11 articles obligatoires** (1128, 1130, 1131, 1132, 1133, 1135, 1137, 1138, 1139, 1112-1, 1240) — tous des principes abstraits du droit des contrats post-réforme 2016. Pattern : **vocabulaire absent (A)** aggravé de **trop d'articles**.

**Fix** : **réécriture LLM impérative** (ajouter dol, réticence, vice du consentement, devoir précontractuel d'information). Sans cela, r@1000 = 0.

### CNB-OBLIG-2025-Q2 — exécution forcée / responsabilité contractuelle

**Q** : Architecte construit étage 10 cm trop bas ; récupérer hauteur ou compensation.

**Top-10 (open et filtered)** : `civil:677` (vues/jours), `urbanisme:R111-28`, `civil:1674` (lésion), `cch:R134-59`, `civil:1793` (forfait architecte). Premier gold core : `1217` (sanctions inexécution) rang 339 open / 72 filtered. `1221` (exécution forcée en nature) rang 10444 / 1678. `1231-1` (dommages-intérêts) rang 12757 / 2033.

**Diagnostic** : BGE-M3 récupère du droit de la construction (1793, urbanisme, vues) — factuellement plausible mais juridiquement à côté. Les articles 1217/1221/1231-1 sont des **articles-principes** du nouveau droit des obligations, jamais cités par leur fait générateur ("10 cm trop bas"). Pattern : **vocabulaire absent + article-principe abstrait (A+B)**.

**Fix** : réécriture LLM (exécution forcée en nature, dommages-intérêts, inexécution contractuelle).

### CNB-OBLIG-2025-Q3 — clause limitative + prescription B2B

**Q** : Mme des Pres / pressing professionnel ; clause "Réclamations" au verso ; délai 30j.

**Top-10 (open)** : entièrement hors-sujet (droit du travail L7423-1, conventions collectives L2261-9, conso L242-7). Filtered : **`code_civil:2254` ← OBLIG rang 1** (aménagement conventionnel de prescription), puis `1245-14, 1724, 1780`. Manquent : `1119` (CGV opposables) rang 341 filtered, `1170` (clause vidant l'obligation essentielle) rang 2386.

**Diagnostic** : `2254` est un succès parfait en filtered (très lexical : "prescription par accord"). Mais `1170` (rang 19521 open / 2386 filtered) est l'article-principe sur les clauses privant l'obligation de sa substance — la question ne dit pas "obligation essentielle", elle décrit le fait. `1119` (CGV opposables si connues/acceptées) — idem. Pattern : **article-principe abstrait (B)**.

**Fix** : recall@10 filtered = 0.33 (1/3 core) déjà acquis ; pour le reste, réécriture LLM ajoutant "clause limitative", "opposabilité des CGV".

### CNB-PROCCIV-2025-Q1 — MARD / ARA

**Q** : Juliette / ex-concubin, 7000 € D&I, recherche solution amiable avec un juge.

**Top-10 (open et filtered identiques)** : `cpc:818, 750, 129-1, 826, 127, civ:840, cpc:820, 750-1` **← EXPERT rang 8**, `821, civ:267`. Trouve `1528` (médiation) rang 133.

**Diagnostic** : **gold_missing critique** — 4/7 articles obligatoires (`1528-1, 1528-2, 1532-1, 1532-2`) **ne sont pas dans le pool LEGI** (décret ARA du 18 juillet 2025, hors snapshot LEGI utilisé). Recall plafonné mathématiquement à 3/7 = 0.43. Le top-10 montre que BGE-M3 récupère correctement le voisinage thématique (conciliation, MARD, 750-1) — c'est cohérent. Pattern : **article hors LEGI**.

**Fix** : rien à faire au niveau retrieval. Il faut **rafraîchir le snapshot LEGI** pour intégrer le décret 2025-715. Sans cela, plafond structurel.

### CNB-PROCCIV-2025-Q2 — appel preuves nouvelles

**Q** : SMS + enregistrements webcam à l'insu, produits en appel.

**Top-10 (open)** : **`cpc:563` ← OBLIG rang 1** ; ensuite `cpp:380-1, cpc:548, conso:R713-6, cpc:1178, 567, 954, cpp:R53-33, postes:L36-8, cpp:441`.

**Diagnostic** : succès sur `563` (très lexical "justifier en appel les prétentions"). Mais `cpc:9` (loyauté de la preuve) rang 3757 open / 1042 filtered ; `civ:9` (vie privée) rang 1193 / 404. r@10 core = 0.33 = 1/3 = plafond car 3 articles gold, 1 trouvé. Pattern : **lexical OK pour le pivot ; vocabulaire absent pour la loyauté de la preuve** (la question décrit "à son insu" mais pas "loyauté"/"licéité").

**Fix** : recall@10 oblig = 0.5 (K*=1). Pour `cpc:9` et `civ:9`, réécriture LLM (loyauté, licéité, vie privée).

### CNB-PROCCIV-2025-Q3 — interruption d'instance / curatelle

**Q** : Christian placé sous curatelle après clôture instruction ; jugement défavorable ; délai appel.

**Top-10 (open et filtered)** : très centrés sur le bon thème — `civ:440, 444`, `cpc:1239`, **`cpc:530` ← OBLIG rang 4**, `civ:433`, `cpc:1249, 1241-1`, `civ:472, 469`. `civ:2241` rang 15 ; `civ:468` rang 140 (rang 66 filtered).

**Diagnostic** : meilleure question civile en termes de qualité top-10. K*_core = 100 (filtered). Mais `cpc:371` (exception : événement après ouverture des débats — **le piège principal** identifié dans la grille) rang 16453 open / 2880 filtered — totalement raté. `cpc:370` (interruption) rang 513 / 200 — moyen. Pattern : **lexical OK partiel ; l'article de l'exception (371) est invisible** parce qu'il dit "Toutefois, l'instance n'est pas interrompue..." — formulation antithétique au signal sémantique de la question.

**Fix** : recall@100 filtered = 0.5 (3/6 oblig). Pour `cpc:371`, c'est un cas pathologique du dense embedding (négation/exception) → multi-vecteur ou BM25 hybride.

## Synthèse de la branche civil

- **Pattern dominant** : **vocabulaire absent / article-principe abstrait** (A+B) — 6 questions sur 9 (CIVIL-Q1, CIVIL-Q2, OBLIG-Q1, OBLIG-Q2, OBLIG-Q3, PROCCIV-Q2). Les cas pratiques CNB décrivent des **faits** ; le gold est constitué d'**articles-principes** post-réforme 2016 (1112-1, 1130, 1217, 1221, 1231-1, 1170…) qui ne contiennent pas le vocabulaire factuel. C'est le mur compositionnel de l'embedding pur.
- **Questions sauvables par fix simple** :
  - Recall@50 filtered : CIVIL-Q1 (partiel), CIVIL-Q3 (oblig 0.5), PROCCIV-Q3 (oblig 0.33).
  - Filtre code strict : CIVIL-Q3 (exclure code_route), CIVIL-Q1 (exclure code_commerce/cch sociétés).
  - = 3 questions partiellement récupérables sans LLM.
- **Questions nécessitant réécriture LLM** : CIVIL-Q2, OBLIG-Q1, OBLIG-Q2, OBLIG-Q3 (compl.), PROCCIV-Q2 (compl.) — **4 à 5 questions**. Aucun fix non-LLM ne fait sortir `1112-1, 1130, 1984, 1217, 1170` du brouillard.
- **Question structurellement plafonnée** : PROCCIV-Q1 (4/7 articles ARA hors LEGI) — refresh corpus requis.
- **Cas positif à étudier** : PROCCIV-Q2 (rang 1 sur l'article pivot) et PROCCIV-Q3 (top-10 entièrement thématique) — BGE-M3 fonctionne quand le gold contient un syntagme lexicalement saillant ("en appel", "curatelle").
- **Recommandation** : (1) pipeline de réécriture LLM (question → expansion juridique abstraite) en pré-requête, c'est la seule manière de débloquer la moitié de la branche civile ; (2) hybride BM25/dense pour les articles à formulation antithétique (cpc:371) ; (3) refresh LEGI pour l'ARA ; (4) filtre code "branche-aware" plus strict pour limiter la code-confusion.
