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
clé en trop).

**Validation des `themes` (résout finding adversarial #1)** — guided decoding
ne contraint que la *forme*, pas la *valeur* : sans contrôle, des variantes
d'orthographe/accent/casse et des paires branche↔sous_branche incohérentes
entreraient comme données « valides » dans 1,12 M records → corruption
silencieuse du filtre. Donc post-génération, **étape de canonicalisation +
validation** (`themes_validation.py`) :

1. Normalisation : trim, casse de phrase, repli accents/espaces, mapping vers
   le libellé canonique le plus proche (table `(branche, sous_branche)` figée
   issue de §5 ; match exact puis fuzzy serré, seuil élevé).
2. Une paire est **acceptée** si elle matche une paire canonique, **ou** si
   elle respecte la regex d'échappatoire `^Autre:[\w \-'’/().]{2,40}$` sur les
   deux niveaux.
3. Paire ni canonique ni `Autre:` bien formée → **paire retirée**, record
   conservé avec `themes_valid:false`, anomalie loggée dans
   `outputs/step1/_themes_anomalies.jsonl` (id + paire brute + raison).

> **Divergence assumée vs reco du reviewer** : il suggérait de traiter une
> paire invalide comme *échec de validation retryable du record entier*. Refusé
> : à 1,12 M appels, jeter les 10 autres champs (extraits correctement, appel
> GPU coûteux déjà payé) à cause d'un seul champ-filtre est un gaspillage net.
> `themes` est un champ d'enrichissement, pas le cœur extractif. **Flag +
> anomalie + canonicalisation** préserve la donnée et trace le drift ; les
> `_themes_anomalies` alimentent la revue d'extension de taxonomie. Le record
> reste pleinement exploitable, le filtre reste propre (paires invalides
> exclues, jamais persistées comme canoniques).

### 3.3 Record JSONL persisté

Champs d'entrée (`id`, `number`, `juris`) + champs de sortie + métadonnées.
**Le record JSONL est la source de vérité unique de la reprise** (cf. §8/§9) :
chaque `id` traité y a exactement un record final avec un `status` terminal.

- `status` : `ok` | `failed_terminal` | `oversized` | `no_fulltext`
  (états **terminaux** uniquement ; un échec *retryable* n'écrit pas de record
  — il part en quarantaine, cf. §9).
- `themes_valid` (bool), `themes_taxonomy_version` (str), `schema_version` (str)
- `model`, `prompt_variant` (cassation|cour_appel|tribunal)
- `tokens_in`, `tokens_out`, `duration_ms`
- `attempt_count` (int), `error_class` (`terminal`|`retryable`|null),
  `error_message` (str|null)

`failed:true` est dérivé de `status != ok` (conservé pour compat. spec Hector).

## 4. Budget de contexte & règle de drop (D4 + conséquence D6)

**Résout finding adversarial #4.** Le `run_all_models.py` du repo passe
`--max-model-len` en CLI avec un **défaut 16384** : supposer 32k en dur ferait
échouer *tardivement* (HTTP 400 vLLM) des arrêts pourtant sous le seuil
théorique. Le budget est donc **vérifié sur le serveur live, pas supposé**.

```
max_model_len   = paramètre de run OBLIGATOIRE (pas de défaut implicite),
                  vérifié au démarrage contre /v1/models du serveur vLLM
                  (échec au démarrage si incohérent — pas par décision)
overhead_prompt = tokens du prompt système ASSEMBLÉ réel
                  (préambule + blocs Hector + BLOC_TAXONOMIE_THEMES + schéma),
                  mesuré une fois au démarrage avec le tokenizer du modèle
SEUIL_input     = max_model_len − max_tokens(4000) − overhead_prompt − marge(512)
```

**Règle de drop** (deux passes, perf + exactitude) :
1. Filtre grossier : `len(fullText)/3.0 > SEUIL_input` → oversized (rapide,
   pas de tokenizer, couvre l'écrasante majorité).
2. **Bande limite** (`±20 %` autour du seuil) : recompte exact via le
   **tokenizer réel du modèle sur le prompt chat entièrement assemblé**
   (système + user) → décision fiable, zéro échec tardif.

Un `id` oversized **n'est pas** mis dans un sidecar : il reçoit un record
JSONL terminal `status:"oversized"` (cf. §3.3/§9) — partie intégrante du
registre idempotent, pas un fichier annexe désynchronisable.

Ordre de grandeur (estimation `/3.0`) : >16k tok = 19 022 JP, >32k tok =
1 575 JP. Avec `max_model_len=32768` le backlog attendu ~5–10k JP (<1 %) ;
avec 16384 il grimpe → **d'où l'obligation de fixer `max_model_len`
explicitement et de le valider au pilote**.

## 5. Taxonomie `themes` (giga-précise, fixe)

Hiérarchie à 2 niveaux **injectée dans le prompt**. Le LLM **doit** choisir
1 à 4 paires `(branche, sous_branche)` de la liste ; à défaut de correspondance
raisonnable, `branche="Autre:<libellé court>"` et
`sous_branche="Autre:<libellé court>"`. Les `Autre:` sont collectées post-hoc
pour **étendre la taxonomie** entre deux runs (bump `themes_taxonomy_version`).

**Source canonique unique** :
[`docs/superpowers/specs/themes-taxonomy-jp.md`](./themes-taxonomy-jp.md) —
**18 branches / ~190 sous-branches**, calée sur la nomenclature « matières »
officielle de la Cour de cassation (endpoint Judilibre `/taxonomy?id=theme`),
affinée pour l'usage filtre. Ce fichier est **gelé** dans
`prompts/step1/themes_taxonomy.py` à l'implémentation (single source of truth ;
le spec ne duplique pas les 190 lignes pour éviter le drift).

### 5.1 Les 18 branches
Droit pénal — fond · Procédure pénale · Droit pénal des affaires · Droit des
obligations et des contrats · Responsabilité civile · Droit des biens et
sûretés · Droit immobilier, baux et construction · Droit de la famille · Droit
des sociétés et des affaires · Entreprises en difficulté · Concurrence,
distribution et propriété intellectuelle · Droit du travail · Sécurité sociale
et protection sociale · Droit de la consommation · Droit des assurances et
bancaire · Procédure civile et arbitrage · Voies d'exécution et juge de
l'exécution · Droit international privé.

### 5.2 Arbitrages structurants (utilisateur, 2026-05-19)
- **Responsabilité contractuelle → branche « Droit des obligations et des
  contrats »** ; « Responsabilité civile » = délictuel + régimes spéciaux
  (produits, Badinter, médical) + réparation. Frontière de filtre la plus
  intuitive.
- **Baux commerciaux → branche « Droit immobilier, baux et construction »**
  (tous les baux regroupés). « Sociétés et affaires » conserve « fonds de
  commerce et opérations sur fonds » ; le multi-thème couvre les arrêts mixtes.
- Tranchés sans escalade (justifiés dans le fichier canonique) : éclatement du
  commercial en 3 branches ; assurances+bancaire fusionnés ; voies d'exécution
  branche autonome / MARD en sous-branche de procédure civile.

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
│   ├── themes_taxonomy.py     # hiérarchie §5 figée + PAIRS canoniques + version + render_for_prompt()
│   └── build_prompt.py        # assemble le prompt final (remplacement explicite, pas .format())
├── schema.py                  # Step1Output (pydantic v2) -> json_schema() pour guided decoding
├── themes_validation.py       # canonicalisation + validation paires (finding #1)
├── budget.py                  # max_model_len live + overhead assemblé + seuil (finding #4)
├── errors.py                  # classification terminal vs retryable (finding #2)
├── ledger.py                  # registre dérivé des sorties + écriture atomique (finding #3)
├── analyzer/
│   └── jp_analyzer.py         # orchestrateur : stream, budget, batch vLLM, parse, valide, persiste, resume, circuit breaker
├── parsing.py                 # parse robuste : json.loads -> strip fence -> jsonrepair
├── run_step1.py               # CLI : --pilot[N] --juris --limit --resume --max-model-len --out
├── serve_vllm.sh              # lance le serveur vLLM gemma4-31B-AWQ (patterns run_all_models.py)
└── tests/                     # tests unitaires (TDD)
outputs/step1/
├── <juris>/part-XXXXX.jsonl   # records terminaux (SOURCE DE VÉRITÉ reprise) — 1 record final/id
├── _quarantine.jsonl          # échecs RETRYABLE non terminaux (id + attempt_count) → re-tentés au prochain run
├── _themes_anomalies.jsonl    # paires thèmes rejetées (finding #1)
└── _metrics.jsonl             # métriques par décision (append, non autoritatif)
```

Plus de `_processed_ids.txt` ni `_backlog_oversized.txt` : ces sidecars
désynchronisables sont supprimés (finding #3). Les `<juris>/part-*.jsonl` sont
écrits **atomiquement** (fichier temp → `fsync` → `rename`) ; le set des `id`
traités est **dérivé au démarrage en scannant les shards JSONL committés**
(seule source de vérité). La quarantaine ne contient que des échecs
*retryable* (jamais d'`id` « définitivement » perdu en silence).

`build_prompt` utilise un **remplacement explicite de placeholders** (pas
`str.format()`) — leçon du bug `KeyError` doctrine_qgen (#18859) : les
accolades JSON du schéma dans le prompt cassent `.format()`.

## 8. Flux d'exécution

0. **Démarrage** : vérifier `max_model_len` live (§4), mesurer l'overhead du
   prompt assemblé, **dériver le set des `id` terminés** en scannant
   `<juris>/part-*.jsonl`, charger la quarantaine (à re-tenter).
1. Stream `jp_index.parquet` row-group par row-group (jamais 5 GB en RAM).
2. Skip si `id` ∈ set terminé dérivé des shards (reprise idempotente).
3. Budget (§4 deux passes) dépassé → record terminal `status:"oversized"`.
4. `text` vide → record terminal `status:"no_fulltext"`.
5. `court ← map(juris)` → `build_prompt(variant)`.
6. Lots traités avec **concurrence client-side** (`--concurrency N`, défaut
   16) : N appels vLLM en vol simultanément via un pool de threads borné
   (vLLM fait du continuous batching côté serveur → la concurrence client est
   le principal levier de débit ; `--concurrency 1` = mode séquentiel).
   Chaque appel : `guided_json=<json_schema>`, `temperature=0.1`,
   `max_tokens=4000`. `analyze_record` est une fonction pure par décision
   donc thread-safe ; circuit breaker et écriture des shards restent
   exécutés dans le thread coordinateur après collecte des futures
   (atomicité §9 préservée).
7. Parse robuste (`json.loads` → strip ```` ```json ```` → `jsonrepair`).
8. Validation Pydantic stricte + canonicalisation `themes` (§3.2).
   - Échec **terminal** (JSON irréparable après 1 retry, contenu invalide) →
     record `status:"failed_terminal"`.
   - Échec **retryable** (timeout/connexion vLLM, 5xx, moteur guided decoding,
     modèle dégradé) → **pas de record terminal** ; `id`+`attempt_count`
     poussés en `_quarantine.jsonl`.
9. Écrire le(s) record(s) terminaux du lot **atomiquement** (temp→fsync→rename
   sur le shard `<juris>/part-*.jsonl`) ; append `_metrics.jsonl`.
10. **Circuit breaker** : taux d'échec (retryable) sur fenêtre glissante
    (défaut : >20 % sur 500 derniers) → **pause/abort du run** avec message
    explicite (évite de marquer des slices entières en échec sur incident
    infra). Isolation par lot sinon : un échec unitaire n'interrompt rien.
11. Reprise/quarantaine : un `id` en quarantaine n'étant pas dans les shards
    terminaux, il est **automatiquement re-tenté** au run suivant jusqu'à
    `max_attempts` (puis record `failed_terminal` avec `error_class:retryable`
    pour traçabilité).

## 9. Gestion d'erreurs & robustesse (résout findings #2 et #3)

**Classification des erreurs (`errors.py`)** — distinction centrale :

| Classe | Exemples | Traitement |
|--------|----------|------------|
| **terminal** | `no_fulltext`, `oversized`, JSON irréparable après 1 retry, contenu structurellement invalide | record JSONL terminal (`status` ad hoc) → entre dans le registre, **jamais re-tenté** |
| **retryable** | timeout/connexion vLLM, HTTP 5xx, erreur moteur guided decoding, modèle dégradé/indisponible | **pas de record terminal** → `_quarantine.jsonl` (`id`, `attempt_count`) → re-tenté au run suivant jusqu'à `max_attempts`, puis bascule `failed_terminal` (`error_class:retryable`, tracé) |

- **Jamais throw vers l'appelant** ; isolation par lot (équiv.
  `Promise.allSettled`). Un échec unitaire n'interrompt ni le lot ni le run.
- **Circuit breaker** : si le taux d'échec *retryable* dépasse un seuil sur
  fenêtre glissante (défaut >20 % / 500), **pause ou abort explicite** du run
  — empêche qu'un incident infra (vLLM mort, env régressé) marque en silence
  des slices entières et les fasse skipper à jamais.
- **Idempotence (finding #3)** : source de vérité = les shards
  `<juris>/part-*.jsonl` écrits **atomiquement** (temp→`fsync`→`rename`). Le
  set des `id` terminés est **dérivé** de ces shards au démarrage (pas de
  sidecar `_processed_ids`/`_backlog` désynchronisable). Dédup garantie : au
  plus un record final par `id` (vérifié à l'agrégation ; en cas de doublon de
  crash, le dernier shard committé prime).
- `max_attempts` (défaut 3), backoff exponentiel sur retryable.
- Métriques par décision : `status`, `error_class`, `attempt_count`, tokens,
  durée, modèle.
- **Pas de gate sur `attendu_cle`** (D2) : champ ni inspecté ni rejeté ; la
  conformité verbatim est portée par le prompt seul.
- Pas de chaîne multi-modèles (un seul gemma ; indisponible → échec *au
  démarrage*, le circuit breaker couvre la dégradation *en cours de run*).

## 10. Plan d'exécution

### Phase Pilote (30 JP stratifiées)
Échantillon : 10 CC + 10 CA + 10 TJ, mélange de tailles (court / médian /
proche du seuil). Objectifs : valider prompts (3 variantes), guided decoding,
parsing, validation schéma, **qualité juridique FR de gemma4-31B**, et
**calage de la taxonomie** (taux de `Autre:`, pertinence des thèmes).
**Benchmark de parallélisation** : rejouer le pilote à `--concurrency` ∈
{1, 8, 16, 32}, mesurer records/min et latence p50/p95, identifier le palier
où le débit sature (le serveur vLLM devient le goulot) → fixe la concurrence
du run complet et l'estimation GPU-heures. Livrable : 30 JSON + rapport
qualitatif + liste des `Autre:` + **tableau débit vs concurrence**.

**Gate** : revue humaine du pilote avant run complet (qualité jugée
acceptable, taxonomie ajustée si besoin).

### Phase Run complet
1 125 968 JP, stream + reprise dérivée des shards, sharding JSONL par juris.
Oversized/no_fulltext/failed = records terminaux dans les mêmes shards.
Circuit breaker actif. Quarantaine re-tentée. Métriques agrégées en fin de run.

## 11. Tests (TDD)

Unitaires (avant implémentation) :
- `step1_routing` : CC→cassation, CA→cour_appel, TJ→tribunal.
- `schema` : record valide accepté ; champ requis manquant rejeté ; clé en
  trop rejetée ; `themes` avec `Autre:` accepté.
- `parsing` : JSON nu OK ; fenced ```` ```json ```` strippé ; JSON cassé
  réparé par `jsonrepair` ; irréparable → erreur explicite.
- `build_prompt` : placeholders remplacés sans `KeyError` même avec accolades
  JSON dans le schéma ; les 3 variantes contiennent le bloc taxonomie.
- `themes_validation` (finding #1) : paire canonique acceptée ; variante
  accent/casse canonicalisée ; `Autre:` bien formé accepté ; paire incohérente
  → retirée + `themes_valid:false` + ligne `_themes_anomalies`.
- `budget` (finding #4) : `max_model_len` absent → erreur au démarrage ;
  incohérent avec `/v1/models` → abort ; record en bande limite recompté au
  tokenizer ; juste au-dessus → `oversized`, juste en dessous → traité.
- `errors` (finding #2) : timeout classé retryable (quarantaine, pas de record
  terminal) ; JSON irréparable classé terminal ; circuit breaker déclenché
  au-delà du seuil de taux d'échec.
- `ledger` (finding #3) : set terminé dérivé des shards ; crash après shard
  committé → pas de doublon à la reprise ; quarantaine re-tentée ; écriture
  temp→rename atomique ; ≤1 record final/`id`.

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

## 15. Revue adversariale Codex (2026-05-19) — résolutions

Verdict initial : `needs-attention` (no-ship). Les 4 findings sont résolus
dans ce spec :

| # | Finding | Résolution | §  |
|---|---------|-----------|----|
| 1 | [high] thèmes seulement contraints par prompt → corruption silencieuse du filtre | `themes_validation.py` : canonicalisation + table de paires figée + regex `Autre:` + `themes_valid` + `_themes_anomalies`. **Divergence assumée** : flag + anomalie au lieu d'échec retryable du record entier (préserve 11 champs coûteux ; justifiée §3.2) | §3.2, §3.3 |
| 2 | [high] échecs retryable deviennent terminaux & skippés à jamais | classification terminal/retryable (`errors.py`), quarantaine re-tentée, `attempt_count`, **circuit breaker** sur taux d'échec | §3.3, §8, §9 |
| 3 | [high] registre de reprise non atomique, oversized non couvert | sidecars supprimés ; source de vérité = shards JSONL **atomiques** (temp→fsync→rename), set terminé **dérivé** ; oversized/no_fulltext = records terminaux ; dédup ≤1/id | §3.3, §7, §8, §9 |
| 4 | [medium] budget contexte supposé 32k vs serveur live (défaut runner=16384) | `max_model_len` paramètre obligatoire vérifié sur `/v1/models` ; overhead = prompt assemblé réel ; bande limite recomptée au tokenizer | §4 |

Re-revue adversariale recommandée après ces changements avant l'implémentation
(le reviewer demandait de bloquer le plan jusqu'à résolution — fait).
