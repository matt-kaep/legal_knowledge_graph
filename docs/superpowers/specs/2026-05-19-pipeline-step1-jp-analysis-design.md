---
date: 2026-05-19
type: decision
tags: [decision, pipeline, jp-analysis, step1, llm, vllm, gemma, knowledge-graph]
status: spec-validé-design
---

# Spec — Pipeline « Step 1 » : analyse structurée de masse des JP Judilibre

## 0. Résumé exécutif

Reconstruire dans ce repo de recherche le pipeline « Step 1 » d'analyse de
jurisprudence d'Hector (origine : `apps/backend/services/jp-analysis`,
TypeScript/OpenRouter), porté en **Python + vLLM sur cluster**, modèle
**open-source `gemma4-31B-AWQ`**, pour produire **un objet JSON structuré par
décision** à partir du seul `fullText` de l'arrêt.

Lecture **neutre et purement extractive** (pas de jugement de pertinence — ça
serait un « Step 2 » hors scope). Corpus : `jp_index.parquet`
(**1 125 968 JP**, colonnes `id, number, juris, text`). Sortie : JSONL sur
disque, persistance DB ultérieure.

Démarrage par un **pilote de 30 JP** stratifiées (smoke test qualitatif),
puis **run complet resumable**.

## 1. Contexte & données

### 1.1 Source

`05-Technique/benchmark/baseline_b2/jp_index.parquet` — 5,0 GB, 24 row-groups,
1 125 968 lignes. Schéma : `id` (ObjectId Judilibre), `number` (n° pourvoi/RG,
**non unique**), `juris`, `text` (texte intégral).

`juris` ∈ **{CC, CA, TJ} uniquement** (corpus 100 % judiciaire) :

| juris | n      | court (spec Hector) | prompt routé      |
|-------|--------|---------------------|-------------------|
| CC    | 553 075| `cc`                | Cassation         |
| CA    | 430 654| `ca`                | Cour d'appel      |
| TJ    | 142 239| `tj`                | Tribunal 1re inst.|

Pas de CE/CAA/TA/Tcom dans ce corpus → **3 prompts exactement**, aucune
ambiguïté de fallback. Pas de colonne `date` ni `ecli` → contrat d'entrée
Hector dégradé sur ces 2 champs (non bloquant : non utilisés en sortie ; la
chambre/formation est ré-inférée par le LLM depuis le `fullText`).

### 1.2 Distribution de taille (mappée le 2026-05-19, corpus complet)

| | p50 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|
| GLOBAL | 2 334 tok | 8 274 | 18 616 | 35 350 | **1 159 959** |
| CC | 1 049 | 4 723 | 13 283 | 28 225 | 386 409 |
| CA | 4 447 | 11 131 | 22 227 | 40 883 | 265 924 |
| TJ | 3 160 | 6 905 | 16 174 | 33 513 | 1 159 959 |

(estimation tokens = `len(chars) / 3.0`, cohérente avec l'analyse pénale du
2026-05-05). Script reproductible : `05-Technique/benchmark/jp_size_distribution.py`.

Distribution très skewée à droite : ~89 % du corpus ≤ 8k tok, mais une queue
extrême (outlier TJ à 1,16 M tok). La queue **n'est pas traitée par ce
pipeline** (cf. §4).

## 2. Décisions de design (verrouillées)

| # | Sujet | Décision |
|---|-------|----------|
| D1 | Fidélité | **Port 1:1 de la spec Hector** + un champ ajouté (`themes`) |
| D2 | `attendu_cle` | **Le prompt garde l'exigence verbatim stricte** (reproduction littérale, blocs Hector inchangés). Tolérance paraphrase = **post-hoc uniquement** : aucun gate, aucun retry, aucune validation bloquante sur ce champ |
| D3 | Sortie JSON | **Guided decoding vLLM (xgrammar)** contraint au JSON Schema |
| D4 | Queue longue | **Drop** si l'arrêt dépasse le budget d'entrée du modèle (cf. §4) → `id` écrit dans un backlog. Traitement ultérieur **hors scope** (batch OpenAI/Claude, côté Matthieu) |
| D5 | `themes` | Champ **ajouté** : hiérarchie `branche → sous_branche` depuis une **taxonomie fixe giga-précise** (§5) + échappatoire `Autre:<libre>` |
| D6 | Modèle | **`gemma4-31B-AWQ`** (`QuantTrio/gemma-4-31B-it-AWQ`, registry `run_all_models.py`). Contexte **32k**. Pilote = validation qualité |
| D7 | Échelle | **Pilote 30 JP stratifiées → run complet** resumable |
| D8 | Params | `temperature=0.1`, `max_tokens=4000`, `timeout` 180 s/appel |

## 3. Contrat d'entrée / sortie

### 3.1 Entrée (par décision, dérivée du parquet)

```
{ id: str, number: str, juris: "CC"|"CA"|"TJ", fullText: str }
```
`court` ← map(`juris`) ; `fullText` vide/absent → record dégradé `failed:true`
sans appel LLM.

### 3.2 Schéma de sortie (JSON strict — 10 champs Hector + `themes`)

```
{
  contexte: str,                       // 1 phrase : juridiction + type de litige
  arguments_parties: [{ partie: str, argument: str, reponse_juge: str }],
  fondements_retenus: str,             // textes + principes retenus
  dispositif: str,                     // ce qui est concrètement décidé
  attendu_cle: str,                    // VERBATIM (prompt strict), 200–1500 ch
  cited_articles: str[],               // "article 1240 code civil", ...
  solution_resume: str,                // 1–2 phrases
  dispositif_summary: str,             // 1–2 phrases
  synthese_pour_avocat: str,           // 2–3 phrases denses, 250–500 ch
  dispositif_nature: str,              // libre, ≥1 ch ("CASSE", "CONFIRME en partie / INFIRME en partie"...)
  themes: [{ branche: str, sous_branche: str }]   // ∈ taxonomie §5, ou "Autre:<libre>"
}
```

Tous champs **requis**, **aucune clé en plus**. Guided decoding garantit la
structure ; validation Pydantic stricte ensuite (rejet si champ manquant /
clé en trop). `themes.branche` et `themes.sous_branche` : type `string`
(pas d'enum figé, à cause de l'échappatoire `Autre:`), valeurs contraintes
**par le prompt** vers la taxonomie §5.

### 3.3 Record JSONL persisté

Champs d'entrée (`id`, `number`, `juris`) + champs de sortie + métadonnées :
`model`, `tokens_in`, `tokens_out`, `duration_ms`, `failed` (bool),
`error_message` (str|null), `prompt_variant` (cassation|cour_appel|tribunal),
`schema_version`.

## 4. Budget de contexte & règle de drop (D4 + conséquence D6)

`gemma4-31B` : contexte **32 768 tokens**. Budget exploitable pour l'arrêt :

```
budget_input = ctx_modèle − max_tokens_sortie − overhead_prompt_système
             ≈ 32768 − 4000 − ~2500  ≈  ~26 000 tokens d'arrêt
```

**Règle de drop** : si `estimate_tokens(fullText) > SEUIL` où
`SEUIL = ctx_modèle − max_tokens − overhead` (calculé, **pas une constante** ;
défaut ≈ 25 000 tok pour gemma 32k) → l'`id` est appendé à
`outputs/step1/_backlog_oversized.txt`, **aucun appel LLM**, on continue.

Ordre de grandeur : >16k tok = 19 022 JP, >32k tok = 1 575 JP. Le backlog
attendu avec gemma se situe entre ces deux bornes (~5–10k JP, <1 %).
`estimate_tokens` = `len(fullText)/3.0` (rapide, conservateur ; pas de
tokenizer chargé dans le filtre).

## 5. Taxonomie `themes` (giga-précise, fixe)

Hiérarchie à 2 niveaux **injectée dans le prompt**. Le LLM **doit** choisir
une paire `(branche, sous_branche)` de la liste ; s'il n'y a aucune
correspondance raisonnable, il met `branche="Autre:<libellé libre court>"` et
`sous_branche="Autre:<libellé libre court>"`. Plusieurs thèmes possibles par
décision (1 à 4 typiquement). Les valeurs `Autre:` sont collectées post-hoc
pour **étendre la taxonomie** entre deux runs.

> Domaine : contentieux **judiciaire** français (civil, commercial, social,
> pénal, et procédures associées). Conçue pour servir de **points d'entrée de
> filtres** stables.

### 5.1 Droit pénal — fond
- Atteintes aux personnes (homicides, violences, mise en danger, harcèlement, viol et agressions sexuelles, atteintes à la dignité)
- Atteintes aux biens (vol, escroquerie, abus de confiance, recel, destruction/dégradation, extorsion)
- Infractions économiques & financières (blanchiment, corruption, fraude fiscale, abus de biens sociaux, délit d'initié)
- Infractions routières (homicide/blessures involontaires routiers, conduite sous l'emprise, défaut de permis/assurance)
- Stupéfiants (usage, détention, trafic, blanchiment associé)
- Atteintes à l'autorité de l'État & terrorisme (association de malfaiteurs, terrorisme, outrage/rébellion)
- Presse & expression (diffamation, injure, provocation)
- Droit pénal du travail / des affaires (travail dissimulé, entrave, mise en danger)
- Peines & sanctions (nature et quantum, sursis, aménagement, confiscation, période de sûreté)
- Responsabilité pénale (imputabilité, complicité, tentative, causes d'irresponsabilité, personnes morales)

### 5.2 Procédure pénale
- Enquête & garde à vue (mesures coercitives, droits de la défense, nullités)
- Instruction (mise en examen, détention provisoire, contrôle judiciaire, expertises, nullités)
- Jugement & audience (citation, comparution, administration de la preuve, droits des parties)
- Voies de recours (appel, pourvoi en cassation, réexamen)
- Action publique & action civile (prescription, constitution de partie civile, transaction/CRPC)
- Exécution des peines (application des peines, libération conditionnelle, contentieux post-sentenciel)
- Entraide & extradition (mandat d'arrêt européen, coopération internationale)

### 5.3 Droit civil — obligations & contrats
- Formation du contrat (consentement, vices, capacité, objet, cause/contenu)
- Exécution & inexécution (responsabilité contractuelle, résolution, exception d'inexécution)
- Régime des obligations (cession, subrogation, compensation, prescription)
- Contrats spéciaux — vente (garantie des vices, conformité, garantie d'éviction)
- Contrats spéciaux — louage & entreprise (contrat d'entreprise, mandat, prêt, dépôt)
- Quasi-contrats (gestion d'affaires, paiement de l'indu, enrichissement injustifié)

### 5.4 Droit civil — responsabilité délictuelle
- Responsabilité du fait personnel (faute, lien de causalité, préjudice)
- Responsabilité du fait des choses & d'autrui (gardien, commettant, parents)
- Responsabilités spéciales (produits défectueux, accidents de la circulation — loi Badinter, troubles anormaux de voisinage)
- Réparation du préjudice (postes de préjudice, évaluation, perte de chance)

### 5.5 Droit des biens & sûretés
- Propriété & démembrements (usufruit, servitudes, indivision, mitoyenneté)
- Possession & publicité foncière (prescription acquisitive, action en revendication)
- Sûretés personnelles (cautionnement, garantie autonome)
- Sûretés réelles (hypothèque, gage, nantissement, privilèges, fiducie-sûreté)

### 5.6 Droit de la famille
- Couple (mariage, PACS, régimes matrimoniaux, divorce, prestation compensatoire)
- Filiation (établissement, contestation, adoption, PMA)
- Autorité parentale & enfance (résidence, contribution à l'entretien, assistance éducative)
- Successions & libéralités (dévolution, réserve, partage, testament, donations)
- Protection des majeurs (tutelle, curatelle, habilitation familiale)

### 5.7 Droit commercial & des affaires
- Sociétés (constitution, gouvernance, responsabilité des dirigeants, pactes, cession de droits sociaux)
- Procédures collectives (sauvegarde, redressement, liquidation, déclaration de créances, responsabilité pour insuffisance d'actif)
- Fonds de commerce & baux commerciaux (cession, déspécialisation, renouvellement, indemnité d'éviction)
- Concurrence & distribution (pratiques restrictives, rupture brutale de relations, clauses de non-concurrence)
- Effets de commerce & instruments de paiement (lettre de change, chèque, garanties bancaires)
- Propriété intellectuelle (marques, brevets, droit d'auteur, concurrence déloyale/parasitisme)
- Contrats commerciaux spéciaux (transport, assurance, agence commerciale, franchise)

### 5.8 Droit du travail & sécurité sociale
- Contrat de travail (formation, requalification CDD/intérim, modification, transfert)
- Rupture (licenciement personnel/économique, rupture conventionnelle, prise d'acte, résiliation judiciaire)
- Conditions de travail (durée du travail, rémunération, santé-sécurité, harcèlement, discrimination)
- Relations collectives (représentation, négociation collective, grève, conflits collectifs)
- Protection sociale (accidents du travail/maladies pro, faute inexcusable, contentieux des prestations)

### 5.9 Procédure civile & voies d'exécution
- Compétence & organisation judiciaire (compétence matérielle/territoriale, litispendance, connexité)
- Action & instance (intérêt/qualité à agir, prescription, péremption, désistement)
- Preuve (charge, modes de preuve, mesures d'instruction, expertise judiciaire)
- Jugement & voies de recours (appel, effet dévolutif, pourvoi, tierce opposition, autorité de chose jugée)
- Procédures spéciales (référé, requête, injonction de payer, procédures accélérées)
- Voies d'exécution & saisies (titre exécutoire, saisie-attribution, saisie immobilière, mesures conservatoires)
- Arbitrage & MARD (clause compromissoire, sentence, médiation/conciliation)

### 5.10 Transverses
- Droit international privé (conflits de lois, conflits de juridictions, exequatur)
- Droit de la consommation (clauses abusives, crédit, démarchage, pratiques commerciales déloyales)
- Droit des assurances (garantie, déchéance, subrogation de l'assureur, assurance de responsabilité)
- Droits & libertés fondamentaux dans le procès (CESDH art. 6, contradictoire, délai raisonnable)
- Responsabilité de la puissance publique devant le juge judiciaire (voie de fait, emprise)

> Cette taxonomie est **validée à l'écriture du spec** puis figée dans
> `prompts/step1/themes_taxonomy.py`. Toute extension passe par revue des
> `Autre:` collectés et un bump de `themes_taxonomy_version`.

## 6. Routage juridiction → prompt (3 variantes, schéma identique)

Chaque prompt final =
`[PRÉAMBULE juridiction] + "\n\n# Règles\n\n" + [BLOC_FACTUEL_PARTAGÉ] + "\n\n" + [BLOC_FORMAT_SORTIE_PARTAGÉ] + [BLOC_TAXONOMIE_THEMES]`.

| juris | Variante | Préambule (repris **verbatim** de la spec Hector §7a/7b/7c) |
|-------|----------|-------------------------------------------------------------|
| CC | Cassation | « …arrêt de la **Cour de cassation**… attendu de principe, visas formels, chambre+formation… » |
| CA | Cour d'appel | « …arrêt de **Cour d'appel**… motifs détaillés, chefs réformés/confirmés, dispositif mixte… » |
| TJ | Tribunal | « …jugement de **tribunal de 1re instance**… solution factuelle, dispositif granulaire… » |

`BLOC_FACTUEL_PARTAGÉ` et `BLOC_FORMAT_SORTIE_PARTAGÉ` : repris **verbatim de
la spec Hector §7d/7e** (y compris l'exigence verbatim stricte de
`attendu_cle` et le calibrage `synthese_pour_avocat`). Seul ajout :
`BLOC_TAXONOMIE_THEMES` (rendu de §5 + consigne `Autre:`).

Messages envoyés :
```
[ {role:"system", content:<prompt routé>}, {role:"user", content:<fullText brut>} ]
```
(user message = texte de l'arrêt **seul**, non wrappé.)

## 7. Architecture

```
05-Technique/benchmark/jp_analysis/
├── prompts/step1/
│   ├── step1_shared.py        # BLOC_FACTUEL_PARTAGÉ, BLOC_FORMAT_SORTIE_PARTAGÉ (verbatim Hector)
│   ├── step1_cassation.py     # préambule CC
│   ├── step1_cour_appel.py    # préambule CA
│   ├── step1_tribunal.py      # préambule TJ
│   ├── step1_routing.py       # juris -> (préambule, variant_name)
│   ├── themes_taxonomy.py     # hiérarchie §5 figée + render_for_prompt()
│   └── build_prompt.py        # assemble le prompt final (remplacement explicite, pas .format())
├── schema.py                  # Step1Output (pydantic v2) -> json_schema() pour guided decoding
├── analyzer/
│   └── jp_analyzer.py         # orchestrateur : stream parquet, filtre budget, batch vLLM, parse, valide, persiste, resume
├── parsing.py                 # parse robuste : json.loads -> strip fence -> jsonrepair
├── run_step1.py               # CLI : --pilot[N] --juris --limit --resume --out
├── serve_vllm.sh              # lance le serveur vLLM gemma4-31B-AWQ (patterns run_all_models.py)
└── tests/                     # tests unitaires (TDD)
outputs/step1/
├── <juris>/part-XXXXX.jsonl   # records de sortie
├── _backlog_oversized.txt     # ids droppés (> budget)
├── _metrics.jsonl             # métriques par décision
└── _processed_ids.txt         # pour reprise idempotente
```

`build_prompt` utilise un **remplacement explicite de placeholders** (pas
`str.format()`) — leçon du bug `KeyError` doctrine_qgen (#18859) : les
accolades JSON du schéma dans le prompt cassent `.format()`.

## 8. Flux d'exécution

1. Stream `jp_index.parquet` row-group par row-group (jamais 5 GB en RAM).
2. Skip si `id` ∈ `_processed_ids.txt` (reprise idempotente).
3. `estimate_tokens(text) > SEUIL` → append `id` au backlog, continue.
4. `text` vide → record `failed:true` (motif `no_fulltext`), continue.
5. `court ← map(juris)` → `build_prompt(variant)`.
6. Accumuler en lots ; appel vLLM batché avec `guided_json=<json_schema>`,
   `temperature=0.1`, `max_tokens=4000`.
7. Parse robuste (`json.loads` → strip ```` ```json ```` → `jsonrepair`).
8. Validation Pydantic stricte ; KO après 1 retry → record `failed:true`
   (`error_message`).
9. Écrire record JSONL + ligne `_metrics.jsonl` + `id` dans `_processed_ids`.
10. Isolation par lot : un échec n'interrompt pas le lot ni le run.

## 9. Gestion d'erreurs & robustesse (spec Hector §8/§9)

- **Jamais throw vers l'appelant** : tout échec → record placeholder
  `failed:true` + `error_message` (audit complet).
- Isolation par lot (équivalent `Promise.allSettled`).
- **Retry** : 1 retry sur échec parse/validation ; pas de chaîne multi-modèles
  (un seul modèle gemma — si gemma indisponible, le run échoue *au démarrage*,
  pas par décision).
- Reprise idempotente via `_processed_ids.txt`.
- Métriques par décision (modèle, tokens, durée, failed, erreur).
- **Pas de gate sur `attendu_cle`** (D2) : on n'inspecte ni ne rejette ce
  champ ; la conformité verbatim est portée par le prompt seul.

## 10. Plan d'exécution

### Phase Pilote (30 JP stratifiées)
Échantillon : 10 CC + 10 CA + 10 TJ, mélange de tailles (court / médian /
proche du seuil). Objectifs : valider prompts (3 variantes), guided decoding,
parsing, validation schéma, **qualité juridique FR de gemma4-31B**, et
**calage de la taxonomie** (taux de `Autre:`, pertinence des thèmes). Livrable :
30 JSON + rapport qualitatif + liste des `Autre:`.

**Gate** : revue humaine du pilote avant run complet (qualité jugée
acceptable, taxonomie ajustée si besoin).

### Phase Run complet
1 125 968 JP, stream + resume, sharding JSONL par juris. Backlog des
oversized écrit en parallèle. Métriques agrégées en fin de run.

## 11. Tests (TDD)

Unitaires (avant implémentation) :
- `step1_routing` : CC→cassation, CA→cour_appel, TJ→tribunal.
- `schema` : record valide accepté ; champ requis manquant rejeté ; clé en
  trop rejetée ; `themes` avec `Autre:` accepté.
- `parsing` : JSON nu OK ; fenced ```` ```json ```` strippé ; JSON cassé
  réparé par `jsonrepair` ; irréparable → erreur explicite.
- `build_prompt` : placeholders remplacés sans `KeyError` même avec accolades
  JSON dans le schéma ; les 3 variantes contiennent le bloc taxonomie.
- filtre budget : `text` > seuil → id au backlog, pas d'appel ; seuil calculé
  depuis `ctx_modèle`.
- reprise : `id` ∈ processed → skip.

Intégration : **le pilote 30 JP est le test d'intégration** (qualité,
choix modèle confirmé, taxonomie calée).

## 12. Prérequis & risques

- **R1 — Env vLLM cluster fragile (bloquant)** : `gemma4-31B` a régressé le
  2026-05-18 (transformers sans archi gemma4, NumPy 2.x — cf. mémoires
  `cluster-user-env-fragile`, obs #18847/#18848). **Prérequis du run** :
  serveur vLLM gemma4-31B-AWQ démarré et un appel test réussi, env pinné
  (`requirements.txt`) revalidé. À vérifier *avant* la phase pilote.
- **R2 — Qualité verbatim `attendu_cle`** : modèle open-weight ; risque de
  paraphrase malgré le prompt strict. Accepté (D2) ; mesuré qualitativement
  au pilote (non bloquant).
- **R3 — Coût GPU run complet** : ~1,12 M appels. Le pilote valide avant
  d'engager les GPU-heures. Estimation de débit à mesurer au pilote.
- **R4 — Guided decoding xgrammar** : compatibilité gemma4-31B-AWQ à
  confirmer au pilote (sinon repli `parse + jsonrepair + retry`).

## 13. Hors scope

- Step 2 (verdict / jugement de pertinence).
- Traitement de la queue > budget (backlog → batch OpenAI/Claude, côté
  Matthieu, ultérieurement).
- Persistance en base de données (sortie = JSONL sur disque ; ingestion DB
  traitée séparément).
- Enrichissement graphe-natif (cited_jp, normalisation clés KG, nœuds
  typés) — écarté (Approche A retenue).

## 14. Références

- Spec source : Hector `apps/backend/services/jp-analysis/` (dump fourni
  par l'utilisateur, 2026-05-19).
- Distribution : `05-Technique/benchmark/jp_size_distribution.py`.
- Infra modèle réutilisée : `05-Technique/benchmark/llm_benchmark/run_all_models.py`
  (MODEL_REGISTRY, alias `gemma4-31B`).
- Leçons réutilisées : bug `.format()`/accolades JSON (#18859), gestion
  gracieuse overflow (#18876), fragilité env cluster (mémoire
  `cluster-user-env-fragile`).
