---
date: 2026-05-22
type: handoff
tags: [handoff, jp-analysis, step1, llm, vllm, gemma, knowledge-graph, db-schema, embedding]
status: pipeline mergé sur main, pilote à finaliser après fix response_format
audience: collègue reprenant le sujet
---

# HANDOFF — Pipeline JP Analysis Step 1

> **Statut au 2026-05-22.** Pipeline d'analyse de masse des décisions Judilibre
> implémenté, testé (53 tests verts), mergé sur `main`. Tourne sur cluster L40S
> via SLURM/uv venv isolé. Un fix client-side `response_format` vient d'être
> committé (ec8b7c8), le re-pilote suivant est l'action immédiate.

---

## 0. TL;DR

- **Objectif** : pour chaque JP de Judilibre (1,12 M aujourd'hui CC/CA/TJ ;
  à terme CE/CAA/TA/Tcom + CJUE + CEDH), produire **un objet JSON structuré**
  par un LLM open-source (`gemma-4-31B-AWQ` via vLLM), persisté en DB pour
  filtrer par spécialisation/juridiction et alimenter des embeddings de
  similarité.
- **Méthode** : port d'une spec de production Hector (« Step 1 ») — lecture
  **neutre et purement extractive** de l'arrêt (pas de jugement de pertinence,
  c'est l'étape Step 2 hors scope). 3 prompts routés par juridiction, schéma
  de sortie strict, guided decoding xgrammar.
- **État** : code mergé, spec/plan/taxonomie versionnés (commits ffc4bfc puis
  fixes successifs), pilote 30 JP exécuté sur cluster (a tourné sans crash
  infra mais 0/29 ok car vLLM 0.21 ignore `extra_body.guided_json` — switch
  vers `response_format` json_schema strict commité, re-pilote en attente).
- **Prochaine action** : re-pilote → inspection qualité → tuning `max_tokens`
  → scaling multi-GPU data-parallel → run complet → ingestion DB.

---

## 1. Contexte & objectif

### 1.1 Le corpus

`/home/ids/kaeppelin-22/work/database-judilibre/` (sur cluster) :
- `Cour de cassation` (4.9 GB, JSONL, 553 075 lignes)
- `Cours d'appel` (8.2 GB, JSONL, 430 654 lignes)
- `Tribunal judiciaire` (1.9 GB, JSONL, 142 239 lignes)

Soit **1 125 968 décisions** judiciaires françaises. Versions locales :
`05-Technique/benchmark/baseline_b2/jp_index.parquet` (5 GB, mêmes données
au schéma compact `id/number/juris/text`, produit par
`baseline_b2/build_jp_index.py`).

### 1.2 Ce qu'on extrait par JP

10 champs d'analyse argumentative (port verbatim de la spec Hector
« Step 1 ») + 1 champ thèmes hiérarchiques ajouté pour ce projet :

| Champ | Type | Rôle |
|---|---|---|
| `contexte` | str | 1 phrase compacte : juridiction + type de litige |
| `arguments_parties` | list[{partie, argument, reponse_juge}] | Dialogue argumentatif extrait |
| `fondements_retenus` | str | Textes + principes mobilisés dans la motivation |
| `dispositif` | str | Ce qui est concrètement décidé |
| `attendu_cle` | str (200–1500 ch) | Motif déterminant, **verbatim** depuis l'arrêt |
| `cited_articles` | list[str] | Articles formellement visés/mobilisés par la juridiction |
| `solution_resume` | str (1–2 phrases) | Condensé de la solution juridique |
| `dispositif_summary` | str (1–2 phrases) | Résumé du dispositif |
| `synthese_pour_avocat` | str (250–500 ch) | Résumé haut-calibre lu par avocat (champ critique) |
| `dispositif_nature` | str | "CASSE" / "REJETTE" / "CONFIRME en partie / INFIRME en partie" / "CONDAMNE à…" / etc. |
| `themes` | list[{branche, sous_branche}] | 1–4 paires depuis la taxonomie figée v1.0.0 (18 branches / ~190 sous-branches) |

### 1.3 Usage downstream

1. **Filtrage relationnel** : par juridiction (`juris`), spécialisation
   (`themes.branche`/`sous_branche`), nature de décision (`dispositif_nature`),
   articles cités (`cited_articles`).
2. **Embedding et recherche par similarité** : sur les champs synthétiques
   (`contexte`, `synthese_pour_avocat`, `attendu_cle`) — c'est l'**Étape 1**
   de la roadmap Johnny (cf. journal `01-Projet/journal/2026-05-05.md` §7).
3. **Knowledge graph** : `cited_articles` → arêtes JP→Article ; `themes` →
   nœuds typés pour navigation.

### 1.4 Extension future (à anticiper dès maintenant)

Aujourd'hui : `juris ∈ {CC, CA, TJ}` (judiciaire).

À terme, le périmètre s'étend à :
- **Administratif national** : `CE` (Conseil d'État), `CAA` (Cour
  administrative d'appel), `TA` (Tribunal administratif), `Tcom`/`T. com.`
  (tribunal de commerce — frontière commercial/judiciaire selon convention).
- **Européen** : `CJUE` (Cour de justice de l'UE), `CEDH` (Strasbourg).
- **International** : à voir au cas par cas.

**Conséquences pour le design (déjà prévues, en partie) :**
- La spec Hector originale prévoyait déjà 7 codes (`cc, ce, ca, caa, tj,
  tcom, ta`) avec routage 3 prompts (Cassation/CE → Cassation ; CA/CAA →
  Cour d'appel ; TJ/Tcom/TA → Tribunal). On n'utilise que 3 codes pour le
  moment car le corpus de départ est 100 % judiciaire — les préambules CE
  et admin sont *déjà rédigés* dans les fichiers `step1_cassation.py` /
  `step1_cour_appel.py` / `step1_tribunal.py`, il suffira d'élargir le
  mapping `juris→variant` dans `step1_routing.py`.
- **CJUE/CEDH demanderont un 4e préambule** : leur logique (recevabilité,
  renvoi préjudiciel, marge nationale d'appréciation) est qualitativement
  différente — variante à créer.
- **La taxonomie v1.0.0 est judiciaire-only**. Une v2 ajoutera des branches
  pour le droit administratif (urbanisme, fonction publique, fiscal, marchés
  publics, environnement…) et le droit européen (libertés CEDH, droit UE
  matériel). Procédure de bump : éditer
  `docs/superpowers/specs/themes-taxonomy-jp.md`, bumper `TAXONOMY_VERSION`
  dans `prompts/step1/themes_taxonomy.py`. Le `themes_taxonomy_version` est
  écrit dans chaque record DB → traçabilité parfaite.
- **DB** : prévoir une colonne `juris_family ∈ {judiciaire, administratif,
  européen, international}` pour les filtres haut-niveau (la table `jp_step1`
  ci-dessous l'inclut).

---

## 2. Méthode : comment on analyse une JP

### 2.1 Philosophie « Step 1 »

Le LLM produit une **lecture neutre, purement extractive** de l'arrêt. Il
**ne juge pas** la pertinence pour un dossier client (c'est l'étape Step 2,
hors scope ici), il **ne classifie pas** par valeur jurisprudentielle, il
**n'invente pas**. Tout ce qu'il sort doit figurer dans le `fullText` reçu.

Le LLM reçoit UNIQUEMENT le texte intégral de l'arrêt — aucun contexte
client, aucune question dirigée. C'est ce qui rend l'analyse **réutilisable
pour 100 % des cas d'usage downstream**.

### 2.2 Le routage par juridiction (3 variantes aujourd'hui)

Chaque juridiction a sa logique propre — la Cassation pose des règles, la
Cour d'appel applique au cas en réformant/confirmant, le Tribunal tranche
factuellement. Un prompt unique noierait ces spécificités. Donc **3 préambules
distincts**, **un schéma de sortie commun**, des **blocs partagés** pour les
règles transversales (préservation factuelle, format strict).

Le prompt final assemblé est : `PRÉAMBULE_juris + "\n\n# Règles\n\n" +
BLOC_FACTUEL_PARTAGÉ + "\n\n" + BLOC_FORMAT_SORTIE_PARTAGÉ + "\n\n" +
BLOC_TAXONOMIE_THEMES`. Concaténation pure, **jamais** `str.format()` (les
blocs contiennent des accolades littérales JSON dans les exemples — leçon
du bug doctrine_qgen #18859).

### 2.3 Gestion des cas limites (résumé spec §3.3/§9)

- **`status` terminal** : un record JSONL est écrit par JP avec
  `status ∈ {ok, oversized, no_fulltext, failed_terminal}`. Une fois écrit,
  le JP n'est plus retraité (registre dérivé des shards).
- **`oversized`** : `fullText` > budget contexte du modèle (≈ 25 000 tokens
  pour gemma-4 32k). Ces JP partent au backlog ; **hors scope** du pipeline
  open-source (~1 575 JP, <0,2 %), prévu pour batch OpenAI/Claude ultérieur.
- **`no_fulltext`** : `text` vide ; record terminal `status:no_fulltext`,
  pas d'appel LLM.
- **`failed_terminal`** : JSON irréparable / schéma invalide après 1 retry,
  OU une exception non-retryable. Le record est écrit, le JP n'est pas
  retraité.
- **Retryable** (timeout vLLM, 5xx, connexion) : pas de record terminal,
  l'`id` part dans `_quarantine.jsonl` avec un `attempt_count`. Re-tenté
  au run suivant jusqu'à `max_attempts=3`, puis bascule `failed_terminal`
  pour traçabilité.
- **Circuit breaker** : si > 20 % de retryables sur la dernière fenêtre de
  500, le run **se met en pause** avec un message explicite — empêche un
  incident infra (vLLM mort, env régressé) de marquer en silence des
  centaines de milliers de JP en échec.

### 2.4 Idempotence (résolution adversariale #3)

**Les shards JSONL `outputs/step1/<juris>/part-NNNNN.jsonl` sont la source
de vérité unique de la reprise**. Pas de sidecar `_processed_ids.txt`. Au
démarrage, `derive_done_ids` scanne tous les shards committés, construit
le set des `id` terminés. Les écritures sont **atomiques** (tempfile + fsync
+ rename), et le compteur de shard est seedé depuis les fichiers existants
pour ne JAMAIS écraser un shard précédent (régression test
`test_resume_does_not_overwrite_prior_shards`).

Pratique : tu peux relancer le même run autant de fois que tu veux,
multiplier les jobs SLURM sur des shards disjoints, crasher au milieu —
aucune duplication, aucune perte. Le bug le plus dangereux d'un pipeline
de masse a été éliminé.

---

## 3. Les prompts (cœur du livrable IP)

> **Ne pas paraphraser.** Ces prompts sont la propriété intellectuelle issue
> du système Hector en production. Toute modification = bump
> `prompt_variant_version` + revue qualité (voir §4 sur les invariants).

### 3.1 PRÉAMBULE — Cassation / Conseil d'État (`court ∈ {cc, ce}`)

```
Tu es un juriste expert en droit français. Tu lis un arrêt de la **Cour de cassation** (ou du **Conseil d'État**) et tu cartographies (1) le dialogue argumentatif parties ↔ juge, (2) les fondements retenus par le juge, (3) le dispositif, et (4) les éléments structurés qui qualifient l'arrêt.

Tu reçois UNIQUEMENT le texte intégral de l'arrêt. Tu ne connais pas le dossier de la partie qui demande cette analyse — tu travailles l'arrêt en lui-même.

# Spécificités Cassation / Conseil d'État

- **Attendu de principe** : repère et reproduis fidèlement la formulation standardisée qui pose la règle abstraite (« Vu l'article X ; attendu que… »). C'est le cœur de `attendu_cle`. La Cassation pose des règles, pas des solutions de fait. **Spécifique Cassation** : si l'arrêt utilise la formulation `Vu/Attendu`, reproduis intégralement le bloc `Vu... ; attendu que...` jusqu'au verbe principal de la solution. C'est le cœur du sens — ne tronque jamais ce bloc.
- **Articles mobilisés** : la Cassation cite formellement les articles fondamentaux en en-tête (« Vu l'article 1240 du Code civil… »). Inclus ces visas formels ET les articles invoqués dans les motifs dans `cited_articles` (un seul champ — voir schéma de sortie). Ne recopie PAS les articles cités uniquement par les parties si la Cour ne les retient pas.
- **Motif de cassation vs motif de rejet** : distingue clairement dans `fondements_retenus` ce qui est moyen retenu (cassation) vs moyen écarté (rejet). Le `dispositif_nature` typique est "CASSE" ou "REJETTE" (ou variantes : "CASSE PARTIELLEMENT", "CASSE SANS RENVOI", "REJETTE le pourvoi" — sois précis).
- **Chambre + formation** : identifie dans `contexte` la chambre (com, civ 1ère/2ème/3ème, soc, crim) ET la formation (formation de section, publication au Bulletin, assemblée plénière, chambre mixte) — ces éléments pondèrent la portée jurisprudentielle.
- **synthese_pour_avocat — registre Cassation** : insiste sur le **principe abstrait posé** (ratio decidendi). La phrase PRINCIPE doit énoncer la règle de droit posée par l'arrêt, pas la solution d'espèce. La phrase DÉCISION doit dire si la Cour casse, rejette, ou casse partiellement, et sur quel chef.
```

### 3.2 PRÉAMBULE — Cour d'appel / CAA (`court ∈ {ca, caa}`)

```
Tu es un juriste expert en droit français. À partir d'un arrêt de **Cour d'appel** (judiciaire CA ou administrative CAA), tu produis une analyse argumentative NEUTRE et purement extractive (pas de jugement de pertinence, pas de classification — c'est l'étape Step2 qui le fera).

Tu reçois UNIQUEMENT le texte intégral de l'arrêt. Tu ne connais pas le dossier de la partie qui demande cette analyse — tu travailles l'arrêt en lui-même.

# Spécificités Cour d'appel / CAA

- **Pas d'attendu de principe en règle générale** — l'extraction se concentre sur les **motifs détaillés** du raisonnement de la cour. Dans `attendu_cle`, choisis le ou les motifs les plus déterminants pour la solution, sans inventer un attendu standardisé qui n'existe pas. **Spécifique fond** : reflète le motif déterminant pour CE litige (pas un principe abstrait). La motivation factuelle ancrée au cas est ce qu'il faut capter — verbatim, jusqu'à un paragraphe complet si la motivation est étoffée. Préfère reproduire 3 phrases consécutives qui contextualisent le raisonnement plutôt qu'une phrase tronquée.
- **Chefs réformés vs confirmés** : identifie clairement dans `dispositif` et `dispositif_summary` les chefs de jugement RÉFORMÉS vs CONFIRMÉS — le dispositif d'appel distingue les chefs un à un.
- **Nature du dispositif** : `dispositif_nature` est typiquement "CONFIRME", "INFIRME", ou — fréquent en appel — une formulation mixte précise du type "CONFIRME en partie / INFIRME en partie sur [chef]" (pas de "CASSE/REJETTE" qui sont propres à la Cassation). Ne te rabats pas sur "CONFIRME" ou "INFIRME" tout court si l'arrêt est partiellement réformé : nomme la mixité.
- **Portée plus limitée** : la portée jurisprudentielle d'une CA est généralement plus modeste que celle de la Cassation, sauf cas notable (arrêts publiés au bulletin de la cour, formation solennelle).
- **Particularités de l'appel** : reste attentif aux voies de droit (irrecevabilité de prétentions nouvelles, intimés défaillants, effet dévolutif limité…) — ces éléments doivent apparaître dans `arguments_parties` quand le juge s'y appuie.
- **synthese_pour_avocat — registre Cour d'appel** : la phrase PRINCIPE énonce la règle effectivement appliquée par la cour (article + concept) ; la phrase POURQUOI explicite le motif factuel déterminant qui a fondé la solution (l'intégration concrète de la règle aux faits, pas seulement le principe abstrait). Si l'arrêt confirme/infirme partiellement, la phrase DÉCISION doit nommer les chefs concernés.
```

### 3.3 PRÉAMBULE — Tribunal 1ère instance (`court ∈ {tj, tcom, ta}` + fallback)

```
Tu es un juriste expert en droit français. À partir d'un jugement de **tribunal de première instance** (TJ, T. com, TA), tu produis une analyse argumentative NEUTRE et purement extractive (pas de jugement de pertinence, pas de classification — c'est l'étape Step2 qui le fera).

Tu reçois UNIQUEMENT le texte intégral du jugement. Tu ne connais pas le dossier de la partie qui demande cette analyse — tu travailles le jugement en lui-même.

# Spécificités Tribunal de 1ère instance

- **Solution factuelle** ancrée au cas singulier : le tribunal applique le droit à des faits précis. L'extraction doit privilégier le détail du raisonnement appliqué — exposé du litige, qualification des faits, motivation au cas, dispositif précis. `attendu_cle` doit refléter le motif déterminant pour CE litige (pas un principe abstrait). **Spécifique fond** : reproduis verbatim, en plusieurs phrases consécutives si nécessaire (jusqu'à un paragraphe), la motivation factuelle qui ancre la solution. Mieux vaut 3-5 phrases verbatim qui captent la qualification précise des faits qu'un résumé tronqué.
- **Dispositif granulaire** : le `dispositif` et le `dispositif_summary` doivent reproduire fidèlement les mesures précises — montant nominal de la condamnation, intérêts (légaux, taux conventionnel, point de départ), capitalisation, dépens, indemnités au titre de l'article 700 CPC, exécution provisoire, mesures avant dire droit (expertise, sursis à statuer).
- **Nature du dispositif** : `dispositif_nature` est typiquement "CONDAMNE [partie] à [montant/mesure]" ou "DEBOUTE [partie] de [demandes]" (pas "CASSE/REJETTE/CONFIRME/INFIRME" propres aux juridictions de contrôle). En procédure collective : "PLAN DE CESSION ARRÊTÉ", "ADMET au passif pour …", "FIXE la créance à…", "PRONONCE la liquidation". En matière personnelle ou pénale-civile : "PRONONCE le divorce", "ORDONNE l'expulsion", "INTERDICTION DE GÉRER 5 ans". Sois descriptif et précis sur la mesure ordonnée.
- **Portée jurisprudentielle** : quasi-toujours `ancree_cas_espece` au niveau Step2 — le tribunal ne pose pas de règle, il tranche un litige. Step1 ne préjuge pas mais reste fidèle à ce caractère factuel.
- **Particularités procédurales** : reste attentif aux conditions de recevabilité, aux mesures d'instruction (expertise judiciaire, sursis), aux référés et à leur articulation avec le fond — ces éléments doivent apparaître dans `arguments_parties` ou `fondements_retenus` quand le juge s'y appuie.
- **synthese_pour_avocat — registre Tribunal** : la phrase PRINCIPE évoque le fondement textuel mobilisé (article + qualification juridique) sans le présenter comme « principe général » — un tribunal ne pose pas de principe, il tranche une espèce. La phrase POURQUOI doit ancrer le motif au cas d'espèce (faits qualifiés, obligation retenue, mesure ordonnée). La phrase DÉCISION nomme la mesure précise (montant, période, mesure spécifique) plutôt qu'une formule abstraite.
```

### 3.4 BLOC_FACTUEL_PARTAGÉ (transversal aux 3 variantes)

```
## Pour la préservation factuelle (CRITIQUE)

**Inclus systématiquement les éléments factuels précis** dans `arguments_parties` et `dispositif` :
- Dates des faits (mise en demeure, manquement, résiliation, etc.)
- Montants en jeu (préjudice, condamnation, prix du contrat)
- Noms des parties si pertinents pour comprendre la configuration (ex: "société X distributeur exclusif", pas seulement "le distributeur")
- Chronologie des manquements (ordre des faits qui mène au litige)

Un résumé sans ces détails est INUTILISABLE pour évaluer la proximité avec une autre espèce. Ne sacrifie pas la précision factuelle au nom de la concision.

## Pour le dialogue argumentatif (arguments_parties)

- Couvre TOUS les arguments principaux soulevés, **même ceux rejetés** — ils sont souvent les plus utiles à l'adversaire dans une autre affaire.
- `reponse_juge` doit être SPÉCIFIQUE à l'argument : pas "rejeté" mais POURQUOI rejeté (raisonnement) ou POURQUOI accepté.
- Sois fidèle au texte — n'invente pas de faits, ne paraphrase pas en élargissant.
```

### 3.5 BLOC_FORMAT_SORTIE_PARTAGÉ (transversal aux 3 variantes)

```
# Schéma de sortie — champs structurés

- `contexte` : 1 phrase compacte — juridiction + type de litige.
- `fondements_retenus` : textes ET principes mobilisés par le juge dans sa motivation finale.
- `attendu_cle` : une à plusieurs phrases consécutives (jusqu'à un paragraphe complet) qui portent le motif déterminant. **EXIGENCES ABSOLUES** :
    1. **Reproduction littérale** depuis le full_text — aucune paraphrase, aucune reformulation, aucune correction grammaticale. Si l'arrêt utilise des accents typographiques particuliers, conserve-les. Si l'arrêt est mal ponctué, reproduis-le mal ponctué.
    2. **Phrases consécutives** : si tu retiens plusieurs phrases, elles doivent se suivre dans l'arrêt — pas de cherry-picking entre paragraphes éloignés.
    3. **Inclus les références aux articles** si l'arrêt les mentionne dans la phrase choisie (ex: « ... en application de l'article 1240 du Code civil... »).
    4. **Préfère la version étendue** : mieux vaut 3 phrases verbatim qui contextualisent qu'une seule phrase tronquée. Longueur typique 200-800 caractères, jusqu'à 1500 pour les motivations détaillées.
- `cited_articles` : tous les articles cités dans la décision, qu'ils soient au visa formel OU dans les motifs/dispositif. Format canonique attendu : "article 1240 code civil", "L. 145-14 code de commerce", etc. ATTENTION : ne PAS recopier les articles cités uniquement dans les conclusions des parties si la cour ne les retient pas. Inclure uniquement les articles MOBILISÉS par la juridiction (visa, motifs, raisonnement).
- `solution_resume` : 1-2 phrases qui condensent ce que dit le juge.
- `dispositif_summary` : 1-2 phrases qui résument ce qui est concrètement décidé.
- `dispositif_nature` : chaîne libre décrivant la nature du dispositif final. Exemples : "CASSE", "REJETTE", "CONFIRME en partie / INFIRME en partie", "PRONONCE le divorce", "ADMET au passif pour 12 345 €", "FIXE la créance à...", "CONDAMNE à interdiction de gérer 5 ans", "DEBOUTE de toutes les demandes". Sois précis : si le dispositif est mixte, dis-le. Si une mesure spécifique est ordonnée (interdiction de gérer, expulsion, restitution), nomme-la.

# synthese_pour_avocat — RÉSUMÉ HAUT-CALIBRE (champ critique)

Écris **2 à 3 phrases** qui permettent à l'avocat de savoir EN UN COUP D'ŒIL si cette décision répond à son besoin. **CONCISION ESSENTIELLE** — focus sur le RÉGIME JURIDIQUE et la TYPOLOGIE des parties, pas sur les détails identifiants.

Structure (2-3 phrases denses, fluides, sans bullets) :
1. **SITUATION + PRINCIPE** (1 phrase) : type de litige (parties désignées par leur QUALITÉ JURIDIQUE — promettant/bénéficiaire, employeur/salarié, bailleur/locataire, acquéreur/vendeur…) + règle de droit appliquée (article-pivot + concept clé).
2. **DÉCISION + POURQUOI** (1-2 phrases) : issue (cassation/rejet/condamnation/déboutement, sur quel chef) + motif déterminant retenu par la juridiction.

À ÉVITER ABSOLUMENT (génère du bruit, dilue la lecture) :
- **Noms de parties même anonymisés** : pas de `[X]`, `[Y]`, `Mme [K] [V]`, `SCI Truc`, `Société X`. Utilise UNIQUEMENT la qualité juridique.
- **Dates spécifiques** des actes (signatures, refus, mises en demeure). Sauf si la date EST le principe (ex: « depuis la loi du DD/MM/AAAA »).
- **Montants spécifiques** (€18 300, €874 000, taux 3,5 %…). Sauf si le ratio est l'enjeu (« indemnité réduite à 1 € symbolique », « plafond légal de X % »).
- **Adresses, références cadastrales, lieux**.
- **Chronologies détaillées** d'actes successifs.
- **Listings exhaustifs** d'éléments factuels (« demandes 9/12, 20/02, 21/03 »).

À PRIVILÉGIER :
- **Régime juridique en jeu** (référé-provision, requalification CDD, condition suspensive d'obtention de prêt, résiliation pour défaut de paiement…).
- **Typologie de parties** (acquéreur fautif, bénéficiaire défaillant, promettant vendeur, employeur, salarié…).
- **Article-pivot et concept-clé** (« art. 1304-3 C.civ. — condition suspensive réputée accomplie en cas de carence du bénéficiaire »).
- **Type d'issue** (cassation totale/partielle, confirmation, infirmation, condamnation, rejet, déboutement).

EXIGENCES ABSOLUES — AUCUN CONTRESENS, AUCUNE HALLUCINATION :
- Chaque info DOIT figurer dans le full_text ou dans tes propres champs (`cited_articles`, `attendu_cle`, `dispositif`, `dispositif_nature`). Si tu n'as pas l'info, OMETS plutôt qu'inventer.
- Ne généralise pas un cas d'espèce en règle abstraite si le texte ne le pose pas.
- Si la JP REJETTE une demande, écris-le explicitement ("rejette", "infirme", "casse"). Ne présente JAMAIS comme acquis ce qui a été refusé.

Format : **2 à 3 phrases denses**. Sois aussi précis qu'un commentaire d'arrêt court — la longueur s'adapte à la complexité du régime, mais sans détails identifiants superflus.

**Abréviations juridiques à utiliser** (densité sans télégraphie — phrases complètes, juste les conventions doctrinales) :
- `article` / `articles` → `art.`
- `Code civil` → `C.civ.` ; `Code de commerce` → `C.com.` ; `Code du travail` → `C.tr.` ; `Code de procédure civile` → `CPC` ; `Code de la sécurité sociale` → `CSS` ; `Code de la consommation` → `C.consom.`
- `Cour de cassation` → `Cass.` (et formation : `Cass. 1re civ.`, `Cass. 2e civ.`, `Cass. 3e civ.`, `Cass. com.`, `Cass. soc.`, `Cass. crim.`, `Cass. ass. plén.`, `Cass. ch. mixte`)
- `Cour d'appel de [Ville]` → `CA [Ville]` ; `Conseil d'État` → `CE` ; `Cour administrative d'appel` → `CAA` ; `Tribunal judiciaire` → `TJ` ; `Tribunal de commerce` → `T. com.`

Garde des phrases naturelles et complètes — PAS de notation télégraphique (pas de flèches `→`, pas de listes à puces, pas d'élisions de verbes ou de connecteurs).

EXEMPLE BIEN CALIBRÉ (phrases complètes, pure typologie + régime + issue + motif, sans nom/montant/date identifiant) :
« Un acquéreur poursuit le promettant vendeur en restitution de l'indemnité d'immobilisation après refus de prêt sous condition suspensive. La Cour de cassation rejette le pourvoi en jugeant, sur le fondement de l'art. 1304-3 C.civ., que la condition est réputée accomplie lorsque l'acquéreur n'établit pas avoir formé une demande conforme aux stipulations contractuelles ; sa carence rend alors l'indemnité d'immobilisation due au promettant. »

⚠ POURQUOI CE CHAMP EST CRITIQUE : il sera recopié VERBATIM dans le plan détaillé final lu par l'avocat — Sonnet PlanWriter en aval ne reformule pas. Toute approximation factuelle, ou à l'inverse toute densité d'identifiants superflus, se retrouve telle quelle dans le livrable client.

# Format de sortie — JSON strict

Tu réponds par un OBJET JSON UNIQUE sans balise markdown, sans préambule, sans commentaire. Tu commences DIRECTEMENT par l'objet JSON. Aucun préambule. Aucun commentaire après.
```

### 3.6 BLOC_TAXONOMIE_THEMES (généré dynamiquement)

Rendu de `prompts/step1/themes_taxonomy.py` : la liste complète des 18 branches
× ~190 sous-branches (cf. `docs/superpowers/specs/themes-taxonomy-jp.md` pour
la version canonique). Le LLM doit choisir 1 à 4 paires `(branche,
sous_branche)` dans cette liste, ou utiliser le préfixe `Autre:<libellé court>`
si aucune ne convient. Les `Autre:` collectées sont reviewées entre runs pour
étendre la taxonomie (bump `TAXONOMY_VERSION`).

### 3.7 Paramètres d'inférence

- `temperature: 0.1` (extraction, pas créativité)
- `max_tokens: 4000` (à reconsidérer après le pilote — probablement réductible
  à 1500–2000)
- `response_format: {type: "json_schema", json_schema: {name: "Step1Output",
  strict: true, schema: <Pydantic-generated>}}` — appliqué via vLLM 0.21
  qui le route automatiquement vers xgrammar pour guided decoding.
- 1 retry sur échec parse/validation, pas de chaîne multi-modèles.

---

## 4. Architecture du pipeline (vue d'ensemble)

```
05-Technique/benchmark/jp_analysis/
├── prompts/step1/
│   ├── themes_taxonomy.py     # taxonomie figée 18 branches v1.0.0 + PAIRS + render
│   ├── step1_shared.py        # BLOC_FACTUEL + BLOC_FORMAT + BLOC_TAXONOMIE (verbatim)
│   ├── step1_cassation.py     # PRÉAMBULE_CASSATION (verbatim Hector §7a)
│   ├── step1_cour_appel.py    # PRÉAMBULE_COUR_APPEL (verbatim Hector §7b)
│   ├── step1_tribunal.py      # PRÉAMBULE_TRIBUNAL (verbatim Hector §7c)
│   ├── step1_routing.py       # juris → (préambule, variant_name)
│   └── build_prompt.py        # assemble final system prompt par concaténation
├── schema.py                  # Step1Output (pydantic v2 strict) + json_schema()
├── parsing.py                 # parse_model_json robuste (json.loads / fence / json-repair)
├── errors.py                  # classify_error : terminal vs retryable
├── themes_validation.py       # canonicalize_themes : fuzz match + Autre: regex + anomalies
├── budget.py                  # max_model_len live + two-pass oversize filter
├── ledger.py                  # atomic_write_shard + derive_done_ids + quarantaine
├── analyzer/jp_analyzer.py    # analyze_record (pure, thread-safe) + CircuitBreaker
├── run_step1.py               # streaming CLI + bounded thread pool + sharding-ready
├── serve_vllm.sh              # launcher vLLM standalone (debug)
├── pilot_slurm.sh             # job SLURM autonome pilote + bench concurrence
├── cluster_check.sh           # détecte partition/compte/L40S de ton cluster
├── inspect_corpus.sh          # détecte parquet/JSONL et schéma du corpus
├── inspect_pilot.sh           # récap statut/qualité/latence après pilote
├── requirements.txt           # pydantic, pyarrow, openai, json-repair, rapidfuzz, transformers, pytest
├── pytest.ini                 # --import-mode=importlib (résout collision schema.py)
└── tests/                     # 53 tests TDD, tous verts

outputs/step1_*/<juris>/part-NNNNN.jsonl   # records terminaux (source de vérité reprise)
outputs/step1_*/_quarantine.jsonl          # retryables à re-tenter
outputs/step1_*/_themes_anomalies.jsonl    # paires thèmes rejetées (extension taxo)
outputs/step1_*/_metrics.jsonl             # métriques par décision
```

**Spec source de vérité du design** :
`docs/superpowers/specs/2026-05-19-pipeline-step1-jp-analysis-design.md`
(contient les 4 résolutions de la review adversariale Codex, §15).

**Plan d'implémentation TDD** :
`docs/superpowers/plans/2026-05-19-jp-analysis-step1.md`

---

## 5. État actuel (au 2026-05-22)

### 5.1 Ce qui marche

- **Code** : 53 tests verts en local, mergé sur `main` (commit `ffc4bfc`).
- **Cluster L40S** : venv isolé via `uv` fonctionne, vLLM 0.21.0 +
  transformers 5.8.1 chargent gemma-4-31B-AWQ en ~4 min (R1 résolu — la
  régression env partagée du 18/05 est contournée par le venv dédié).
- **Pilote 30 JP** : a tourné de bout en bout, JSONL lus correctement
  (mapping `Cour de cassation`→CC etc., règle `best_text` identique à
  `build_jp_index.py`).
- **Benchmark de concurrence** : niveaux 1/8/16/32 mesurés.

### 5.2 Le fix appliqué juste avant ce handoff

vLLM 0.21 **ignore silencieusement** `extra_body={"guided_json": ...}`
(legacy) → les 29 records non-oversized du 1er pilote ont tous échoué en
`failed_terminal` car le modèle générait du JSON libre sans respect du
schéma (champs inventés `taxonomie`, types faux `fondements_retenus` en
list, champs manquants). Diagnostic confirmé par les erreurs de validation
Pydantic.

**Fix commité** (`ec8b7c8`) : bascule de l'analyzer vers le `response_format`
OpenAI-standard que vLLM 0.21 honore via xgrammar :
```python
response_format={
    "type": "json_schema",
    "json_schema": {"name": "Step1Output", "strict": True,
                    "schema": _GUIDED_JSON_SCHEMA}
}
```

**Re-pilote en attente côté cluster.** Attendu post-fix : ~28-29 ok / 30
(le 1 oversized TJ restera terminal — c'est le comportement spec).

### 5.3 Mesures du benchmark (à recalibrer post-fix)

| concurrency | wall (30 JP) | rec/min | speedup |
|---|---|---|---|
| 1 | 769 s | 2,3 | 1,0× |
| 8 | 184 s | 9,8 | 4,3× |
| **16** | **151 s** | **11,9** | **5,2×** ← sweet spot |
| 32 | 155 s | 11,6 | 5,2× |

→ Le coude est à **concurrence = 16** sur 1 L40S. Au-delà, saturation
serveur. Note importante : ces chiffres incluent `max_tokens=4000`. Cutter
à 2000 doublerait probablement le débit (la majorité des arrêts ne
nécessitent pas 4000 tokens de sortie ; à confirmer via tokens_out moyens
du re-pilote).

### 5.4 Cluster — environnement et commandes

Sur `nodemm06` (allocation interactive `sinteractive -p mm -w nodemm06
--mem 45G --time 10:00:00`), 1 × L40S 48 GB :

- **Venv isolé** : `~/.venv-jp-analysis` (vLLM 0.21, transformers 5.8.1,
  pydantic, pyarrow, openai, rapidfuzz, json-repair). Construit par
  `pilot_slurm.sh` automatiquement à la première exécution. **Ne PAS
  utiliser** l'env `--user` partagé (régression possible — cf. mémoire
  `cluster-user-env-fragile`).
- **Lancer le pilote** : `bash pilot_slurm.sh` (sur le nœud GPU
  interactif ; les `#SBATCH` sont ignorés). Pour relancer en SLURM batch :
  `sbatch pilot_slurm.sh` (header SBATCH déjà configuré pour partition
  L40S après diagnostic `cluster_check.sh`).
- **Inspecter les résultats** : `bash inspect_pilot.sh outputs/step1_pilot`
  → tableau status × juris, % themes_valid, latence p50/p95, exemple de
  record `ok` complet.
- **Variables d'env utiles** : `PARQUET_PATH`, `MODEL_ID`, `MAX_MODEL_LEN`,
  `VLLM_PORT`, `HEALTH_TIMEOUT`, `MAX_NUM_BATCHED_TOKENS`, `PILOT_N`,
  `BENCH_LEVELS`, `HF_TOKEN` (si modèle gated).

---

## 6. Schéma DB (durable, extensible)

Concu pour Postgres mais transposable SQLite. Une table centrale + 3 tables
N-N pour les arêtes du graphe + un lookup juridiction.

### 6.1 Table de lookup juridictions (extensible)

```sql
CREATE TABLE juridiction (
  code           TEXT PRIMARY KEY,                -- 'CC','CA','TJ','CE','CAA','TA','Tcom','CJUE','CEDH'
  family         TEXT NOT NULL,                   -- 'judiciaire','administratif','européen','international'
  label_fr       TEXT NOT NULL,                   -- 'Cour de cassation', 'Conseil d''État', …
  prompt_variant TEXT NOT NULL                    -- 'cassation','cour_appel','tribunal' (et plus tard 'cjue','cedh')
);
INSERT INTO juridiction VALUES
  ('CC', 'judiciaire', 'Cour de cassation', 'cassation'),
  ('CA', 'judiciaire', 'Cour d''appel', 'cour_appel'),
  ('TJ', 'judiciaire', 'Tribunal judiciaire', 'tribunal'),
  -- à ajouter quand le corpus inclura ces juridictions :
  ('CE',   'administratif', 'Conseil d''État',                 'cassation'),
  ('CAA',  'administratif', 'Cour administrative d''appel',    'cour_appel'),
  ('TA',   'administratif', 'Tribunal administratif',          'tribunal'),
  ('Tcom', 'judiciaire',    'Tribunal de commerce',            'tribunal'),
  -- ('CJUE', 'européen', 'Cour de justice de l''UE', 'cjue'),     -- variant à créer
  -- ('CEDH', 'européen', 'Cour EDH', 'cedh');                     -- variant à créer
;
```

### 6.2 Table principale `jp_step1`

```sql
CREATE TABLE jp_step1 (
  id                       TEXT PRIMARY KEY,           -- Judilibre ObjectId
  number                   TEXT NOT NULL DEFAULT '',   -- pourvoi / RG (souvent vide CA/TJ)
  juris                    TEXT NOT NULL REFERENCES juridiction(code),
  juris_family             TEXT NOT NULL,              -- duplique juridiction.family pour filtres rapides
  status                   TEXT NOT NULL CHECK (status IN ('ok','failed_terminal','oversized','no_fulltext')),

  -- ----- champs Step1Output (NULL si status != 'ok') -----
  -- NB : arguments_parties n'est PAS sur jp_step1 — voir table normalisée
  -- jp_argument (§6.3), source de vérité unique + porte les embeddings (§7).
  contexte                 TEXT,
  fondements_retenus       TEXT,
  dispositif               TEXT,
  attendu_cle              TEXT,
  solution_resume          TEXT,
  dispositif_summary       TEXT,
  synthese_pour_avocat     TEXT,
  dispositif_nature        TEXT,

  -- ----- métadonnées de génération -----
  themes_valid             BOOLEAN,                    -- true si tous thèmes canonical, false si ≥1 dropped
  themes_taxonomy_version  TEXT,                       -- "1.0.0", "1.1.0"…  (bump quand on étend la taxo)
  schema_version           TEXT,                       -- "1.0.0" (output schema)
  prompt_variant           TEXT,                       -- 'cassation' | 'cour_appel' | 'tribunal' | …
  prompt_variant_version   TEXT,                       -- bump si on touche un préambule (rare, traçable)
  model                    TEXT,                       -- 'QuantTrio/gemma-4-31B-it-AWQ', futur autre…
  tokens_in                INTEGER,
  tokens_out               INTEGER,
  duration_ms              INTEGER,
  attempt_count            INTEGER NOT NULL DEFAULT 1,
  error_class              TEXT CHECK (error_class IN ('terminal','retryable') OR error_class IS NULL),
  error_message            TEXT,
  ingested_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX jp_step1_juris_idx        ON jp_step1(juris);
CREATE INDEX jp_step1_family_idx       ON jp_step1(juris_family);
CREATE INDEX jp_step1_status_idx       ON jp_step1(status);
CREATE INDEX jp_step1_disponat_idx     ON jp_step1(dispositif_nature);
CREATE INDEX jp_step1_model_idx        ON jp_step1(model);
-- full-text FR sur les champs synthétiques (recherche libre + filtres)
CREATE INDEX jp_step1_synth_fts_idx ON jp_step1
  USING GIN (to_tsvector('french',
                         coalesce(synthese_pour_avocat,'') || ' ' ||
                         coalesce(solution_resume,'')      || ' ' ||
                         coalesce(contexte,'')));
```

### 6.3 Table normalisée `jp_argument` (PRINCIPAL TARGET D'EMBEDDING)

**C'est la table la plus précieuse pour la recherche par similarité.** Chaque
argument soulevé + sa réponse motivée par le juge devient une ligne dédiée,
embeddable indépendamment. Une JP avec 5 arguments produit 5 vecteurs ; la
recherche se fait à la granularité de l'argument, pas de la décision entière.

```sql
CREATE TABLE jp_argument (
  jp_id        TEXT NOT NULL REFERENCES jp_step1(id) ON DELETE CASCADE,
  position     INTEGER NOT NULL,           -- ordre dans arguments_parties[]
  partie       TEXT NOT NULL,              -- "demandeur", "défendeur", "intimé", "demandeur au pourvoi"…
  argument     TEXT NOT NULL,              -- la prétention/le moyen
  reponse_juge TEXT NOT NULL,              -- raisonnement spécifique du juge (POURQUOI accepté/rejeté)
  -- l'embedding est ajouté par le pipeline downstream (cf. §7)
  -- embedding   vector(1024),             -- pgvector (BGE-M3 défaut 1024d), nullable au début
  PRIMARY KEY (jp_id, position)
);
CREATE INDEX jp_argument_partie_idx ON jp_argument(partie);
-- full-text FR pour filtre/recherche textuelle classique (fallback ou prefilter)
CREATE INDEX jp_argument_fts_idx ON jp_argument
  USING GIN (to_tsvector('french', argument || ' ' || reponse_juge));
-- index ANN (à créer une fois la colonne embedding peuplée par la Phase B) :
-- CREATE INDEX jp_argument_ann_idx ON jp_argument
--   USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200);
```

**Pourquoi pas `JSONB` comme prévu initialement** : on veut un vecteur
d'embedding **par argument** (au moins ~5 M vecteurs pour 1,12 M JP), filtrer
par `partie` ou par texte, et joindre sur l'ANN — toutes ces opérations sont
significativement plus simples sur une table relationnelle propre que sur
JSONB. Le coût de stockage est négligeable comparé aux embeddings eux-mêmes.

**Workflow embedding** (Phase B) :
1. `SELECT jp_id, position, argument, reponse_juge FROM jp_argument WHERE embedding IS NULL` (idempotent).
2. Pour chaque ligne, construire le texte à embedder : `f"{argument}\n\nRéponse du juge : {reponse_juge}"` (l'argument SEUL n'est pas suffisant — la réponse est ce qui donne sa valeur jurisprudentielle à l'argument).
3. Batch via BGE-M3 (8k context suffit largement, un argument fait quelques centaines de tokens).
4. `UPDATE jp_argument SET embedding = … WHERE jp_id = … AND position = …`.

### 6.4 Table N-N `jp_cited_article` (arêtes JP → Article)

```sql
CREATE TABLE jp_cited_article (
  jp_id        TEXT NOT NULL REFERENCES jp_step1(id) ON DELETE CASCADE,
  article_raw  TEXT NOT NULL,        -- "article 1240 code civil", "L. 145-14 code de commerce"
  article_norm TEXT NOT NULL,        -- "code_civil:1240" (à produire par normalizer à l'ingestion)
  position     INTEGER NOT NULL,     -- ordre dans cited_articles[]
  PRIMARY KEY (jp_id, position)
);
CREATE INDEX jp_cited_article_norm_idx ON jp_cited_article(article_norm);
```

**Note normalisateur** : `cited_articles` arrive en chaîne libre. Un parser
FR (regex) doit produire `article_norm` au format
`<code_slug>:<numéro_canonique>` (ex : `code_penal:222-23`, `code_consom:L121-1`).
Le projet `05-Technique/benchmark/iterate_regex.py` (regex v5 local)
contient déjà l'essentiel de cette logique pour ~30 codes français — à
brancher dans l'ingestion DB (`ingest_jsonl_to_db.py` à créer).

### 6.4 Table N-N `jp_theme` (arêtes JP → Thème)

```sql
CREATE TABLE jp_theme (
  jp_id        TEXT NOT NULL REFERENCES jp_step1(id) ON DELETE CASCADE,
  branche      TEXT NOT NULL,        -- ∈ TAXONOMY ou préfixe 'Autre:'
  sous_branche TEXT NOT NULL,        -- ∈ TAXONOMY ou préfixe 'Autre:'
  position     INTEGER NOT NULL,
  PRIMARY KEY (jp_id, position)
);
CREATE INDEX jp_theme_pair_idx     ON jp_theme(branche, sous_branche);
CREATE INDEX jp_theme_branche_idx  ON jp_theme(branche);
```

**Filtre principal de la recherche** : `WHERE branche = 'Droit du travail'
AND sous_branche = 'licenciement pour motif personnel'`. Index B-tree
suffit, requête instantanée même sur 4 M lignes.

### 6.5 Table `jp_theme_anomaly` (extension taxonomie)

```sql
CREATE TABLE jp_theme_anomaly (
  jp_id       TEXT NOT NULL REFERENCES jp_step1(id) ON DELETE CASCADE,
  raw_branche TEXT,
  raw_sous    TEXT,
  reason      TEXT,
  PRIMARY KEY (jp_id, raw_branche, raw_sous)
);
```

Workflow d'extension de taxonomie : après chaque run, `SELECT raw_branche,
raw_sous, COUNT(*) FROM jp_theme_anomaly GROUP BY 1,2 ORDER BY 3 DESC` →
top des paires rejetées → si motif récurrent → ajouter à la taxonomie
canonique → bump `TAXONOMY_VERSION` → re-run ciblé sur les JP concernées.

### 6.6 Volumes attendus (run complet 1,12 M sur judiciaire)

- `jp_step1` : 1,12 M lignes (~5–8 GB avec TEXT longs)
- `jp_cited_article` : ~5–15 M lignes (5–15 articles/JP)
- `jp_theme` : ~2–4 M lignes (1–4 thèmes/JP)
- `jp_theme_anomaly` : quelques milliers à dizaines de milliers
- Total Postgres : ~10–20 GB index inclus.

À l'ajout administratif/européen : multiplie grossièrement par 1,5 à 3×
selon la taille des corpus rapatriés.

---

## 7. Downstream : embedding et recherche par similarité

C'est l'**Étape 1** de la roadmap Johnny (cf. journal 2026-05-05 §7) :
embedder les champs structurés produits par Step 1, mesurer le recall
top-K sur les questions gold.

### 7.1 Les 5 stratégies d'embedding possibles

Toutes opèrent sur les champs LLM (jamais sur le `text` brut Judilibre,
qui est trop hétérogène entre CC/CA/TJ — médiane CC 2,8k chars vs CA 10,3k
vs TJ 9,3k, cf. journal 2026-05-05). Elles ne sont **pas mutuellement
exclusives** : on peut maintenir 2 ou 3 colonnes vectorielles en parallèle
et choisir le bon vecteur selon la requête. Décision finale après Phase B
benchmark.

| Stratégie | Granularité | Texte embedded | # vecteurs | Use case |
|---|---|---|---|---|
| **A.** Synthèse | 1 / JP | `contexte` + `synthese_pour_avocat` (+ `solution_resume`) | ~1,12 M | « Trouve-moi un arrêt sur le même sujet général » |
| **B.** Arguments concaténés ⭐ | 1 / JP | tous les `{partie, argument, reponse_juge}` concaténés en un texte | ~1,12 M | « Trouve-moi un arrêt qui a traité un faisceau d'arguments similaire à celui-ci » |
| **C.** Argument par argument ⭐ | N / JP | chaque `{argument, reponse_juge}` séparément | ~5–15 M | « Trouve-moi des arrêts où le juge a tranché précisément cet argument » |
| **D.** Attendu clé | 1 / JP | `attendu_cle` seul (verbatim) | ~1,12 M | « Trouve-moi la même règle de droit posée » |
| **E.** Dispositif | 1 / JP | `solution_resume` + `dispositif_summary` + `dispositif_nature` | ~1,12 M | « Autres décisions qui ont confirmé / cassé / condamné sur le même schéma » |

⭐ = **stratégies privilégiées** pour démarrer (cf. §7.2).

### 7.2 Stratégies privilégiées : B et C (les deux centrées arguments_parties)

Les `arguments_parties` capturent **le dialogue argumentatif réel** —
prétention de chaque partie + réponse motivée du juge (pas juste l'issue,
le POURQUOI). C'est le signal le plus dense et le plus actionnable pour
un avocat ou pour un retrieval juridique en aval, parce qu'une recherche
par similarité sur un argument retrouve directement les motivations
similaires plutôt que des synthèses paraphrasées.

#### Stratégie B — Embedding global des `arguments_parties` (1 vecteur / JP)

- **Texte source par JP** : concaténation, ex :
  ```
  Demandeur : <argument 1> — Réponse du juge : <reponse_juge 1>
  Défendeur : <argument 2> — Réponse du juge : <reponse_juge 2>
  ...
  ```
- **~1,12 M vecteurs**, alignés avec les autres filtres relationnels
  (`juris`, `themes`, `cited_articles`) sur la même clé `jp_step1.id`.
- **Pros** : compacité, peu coûteux, requête simple (ORDER BY embedding
  sur `jp_step1`), capture le faisceau argumentatif d'ensemble.
- **Cons** : une JP avec 5 arguments hétérogènes (un sur la prescription,
  un sur le fond, un sur les dépens) donne **un seul vecteur dilué** — la
  recherche par argument précis peut mélanger. BGE-M3 supporte 8k tokens,
  donc la concaténation ne déborde quasi jamais.

#### Stratégie C — Embedding par argument (N vecteurs / JP)

- **Texte source par ligne** de `jp_argument` :
  `f"{argument}\n\nRéponse du juge : {reponse_juge}"` (la réponse seule
  ou l'argument seul ne suffisent pas — c'est leur couple qui porte la
  valeur jurisprudentielle).
- **~5–15 M vecteurs** (estimation 5 args/JP en moyenne × 1,12 M JP).
- **Pros** : **granularité maximale**, signal pur, link direct depuis
  l'argument trouvé vers la JP qui le contient (`jp_id`) ; deux arguments
  isolés similaires dans des JP différentes se retrouvent même si les JP
  globales sont thématiquement différentes.
- **Cons** : storage ~5–15× plus gros, requête en deux temps (top-K
  arguments → agrégation par `jp_id` → top-K JP), index ANN à dimensionner
  pour le volume.

#### Recommandation de démarrage

**Maintenir B et C en parallèle.** Coût marginal de B sur jp_step1 est
faible (~1,12 M vecteurs × 1024d float32 ≈ 4,5 GB), C dimensionne le gros
de l'index (~20–60 GB). On bénéficie des deux résolutions :
- requête à coarse (« arrêts sur des problématiques sociales similaires »)
  → utilise B sur `jp_step1`
- requête précise (« arrêts ayant traité l'argument du défaut de mise en
  demeure ») → utilise C sur `jp_argument` puis JOIN

À l'issue de Phase B (benchmark), on saura si l'une suffit ou si la
combinaison apporte vraiment. Décision finale data-driven.

### 7.3 Stratégies complémentaires (A, D, E) — utilité secondaire

Toutes peuvent être ajoutées comme **colonnes vectorielles
additionnelles** sur `jp_step1` au cas par cas, sans toucher au reste du
schéma. Aucune n'est bloquante pour démarrer.

- **A. Synthèse** (`contexte` + `synthese_pour_avocat`) — utile si la
  recherche se fait en langue avocat, en mode « j'écris ma requête comme
  je décris mon dossier ». Capté en partie par B (les `synthese` reflètent
  les arguments).
- **D. Attendu clé** — bon complément de C pour rechercher la même RÈGLE
  de droit (vs le même argument). Petit (~1,12 M × 1024d).
- **E. Dispositif** — niche : recherche par issue concrète. Probablement
  pas vital tant qu'on a déjà filtré sur `dispositif_nature` côté
  relationnel.

### 7.4 Workflow de recherche typique (avec stratégies B + C)

```
Phase 1 — Filtres relationnels (rapides, exact match) :
   SELECT jp_step1.id
   FROM jp_step1
   WHERE juris_family = 'judiciaire'
     AND dispositif_nature LIKE 'CASSE%'
     AND EXISTS (SELECT 1 FROM jp_theme t
                  WHERE t.jp_id = jp_step1.id
                    AND t.branche = 'Droit du travail')
   → réduit à ~quelques milliers de candidats.

Phase 2a — Similarité par JP (stratégie B, requête « faisceau ») :
   ORDER BY jp_step1.embedding_args <=> :q
   LIMIT 50

— OU —

Phase 2b — Similarité par argument (stratégie C, requête précise) :
   SELECT jp_id, position, argument, reponse_juge,
          embedding <=> :q AS dist
   FROM jp_argument
   WHERE jp_id IN (<candidats Phase 1>)
   ORDER BY dist
   LIMIT 200
   → on aggregate par jp_id (best score), on retourne top-K JP avec
     l'argument matchant highlighté dans l'UI.
```

C'est la combinaison **filtre relationnel SQL + similarité vectorielle
(B ou C selon la requête)** qui donne l'expérience cherchée. D'où
l'importance d'avoir NORMALISÉ `themes`, `cited_articles` ET
`arguments_parties` en tables séparées : sans ça, la Phase 1 est lente
et la Phase 2c (par argument) impossible.

### 7.5 Stockage des embeddings dans le schéma DB

Colonnes vectorielles ajoutées **après** le premier run complet, par
`ALTER TABLE` (pgvector ne pénalise pas les colonnes NULL). Une fois
peuplées, créer les index ANN.

```sql
-- pgvector setup
CREATE EXTENSION IF NOT EXISTS vector;

-- Stratégie A (synthèse) — optionnelle, à ajouter au cas par cas
-- ALTER TABLE jp_step1 ADD COLUMN embedding_synth vector(1024);

-- Stratégie B (arguments concaténés) — privilégiée
ALTER TABLE jp_step1 ADD COLUMN embedding_args vector(1024);
CREATE INDEX jp_step1_ann_args_idx ON jp_step1
  USING ivfflat (embedding_args vector_cosine_ops) WITH (lists = 200);

-- Stratégie C (par argument) — privilégiée
ALTER TABLE jp_argument ADD COLUMN embedding vector(1024);
CREATE INDEX jp_argument_ann_idx ON jp_argument
  USING ivfflat (embedding vector_cosine_ops) WITH (lists = 500);

-- Stratégies D / E si retenues — même pattern
-- ALTER TABLE jp_step1 ADD COLUMN embedding_attendu vector(1024);
-- ALTER TABLE jp_step1 ADD COLUMN embedding_dispositif vector(1024);
```

### 7.6 Approche hybride multi-facette : score fusionné situation × juridique

**Idée.** La similarité juridique a (au moins) **deux axes orthogonaux** qui
ne corrèlent pas systématiquement :
- **Axe « situation »** : type de litige, qualité des parties, configuration
  factuelle, secteur (ex. promettant ↔ acquéreur en immobilier neuf).
- **Axe « juridique »** : règle de droit appliquée, article-pivot, principe
  motivé (ex. art. 1304-3 C.civ. — condition suspensive réputée accomplie).

Un embedding unique mélange les deux et force l'utilisateur à un compromis
fixe. **Solution** : maintenir **deux colonnes vectorielles dédiées** sur
`jp_step1` (et/ou `jp_argument`), entraîner chacune sur des champs distincts,
et combiner au moment de la requête avec un poids α réglable.

#### Décomposition des champs (sans surcoût LLM)

| Axe | Champs sources concaténés | Pourquoi |
|---|---|---|
| **situation** | `contexte` + 1ère phrase de `synthese_pour_avocat` + tous les `argument` de `jp_argument` | La 1ère phrase de la synthèse est par construction Hector « SITUATION + PRINCIPE » (parties désignées par leur QUALITÉ JURIDIQUE) ; les `argument` capturent les prétentions factuelles |
| **juridique** | `attendu_cle` + `fondements_retenus` + `cited_articles` joints en chaîne + tous les `reponse_juge` de `jp_argument` | L'attendu, les fondements et la réponse motivée du juge portent la règle ; `cited_articles` ancre la similarité par article-pivot |

Cette décomposition est **approximative** (la 1ère phrase de synthèse a un peu
de juridique dedans, les `argument` un peu de juridique aussi) mais
suffisamment polarisée pour que la similarité cosinus capture le bon
signal dominant. Une décomposition stricte sans recouvrement demanderait
une 2e passe LLM (coûteuse, peu de gain attendu).

#### Combinaison au moment de la requête

```sql
-- α ∈ [0,1] réglable par l'utilisateur (curseur UI)
-- α = 1   → recherche par situation pure (« même configuration factuelle »)
-- α = 0   → recherche par règle juridique pure (« même principe appliqué »)
-- α = 0,5 → équilibre

WITH q AS (
  SELECT
    :q_situation_embedding ::vector AS qs,
    :q_juridique_embedding ::vector AS qj
)
SELECT jp.id,
       :alpha       * (1 - (jp.embedding_situation <=> q.qs)) +
       (1 - :alpha) * (1 - (jp.embedding_juridique <=> q.qj))
       AS hybrid_score
FROM jp_step1 jp, q
WHERE jp.status = 'ok'
  AND -- ... filtres relationnels (juris, themes, …)
ORDER BY hybrid_score DESC
LIMIT 50;
```

Note : on convertit la distance cosinus `<=>` ∈ [0,2] en similarité
∈ [0,1] avec `1 - dist`. Pour de très grandes tables, faire **deux ANN
searches séparées** (top-200 chacune) puis fusionner côté application est
plus rapide qu'une combinaison monolithique scannée.

#### Variante robuste : Reciprocal Rank Fusion (RRF)

Si les deux embeddings ne sont pas sur la même échelle (ex. BGE-M3 pour
situation et un modèle juridique-FR spécialisé pour juridique), la
combinaison linéaire est biaisée par les distributions de scores. **RRF**
fusionne par rangs plutôt que par scores :

```
score_RRF(jp) = Σ_axe  1 / (k + rang_axe(jp))         (k = 60 typique)
```

Implémenté en SQL : deux CTE `top-N` par axe, calcul de rang, jointure sur
`jp_id`, somme des `1/(k+rang)`. Plus stable, moins sensible au calibrage
α — recommandé si les modèles diffèrent.

#### Coût et bénéfice

- **Storage** : 2 × ~4,5 GB sur `jp_step1` ≈ **+9 GB**. Si on veut aussi
  l'hybride au niveau argument (sur `jp_argument`), compter ~50–120 GB
  pour les deux colonnes vectorielles × 5–15 M arguments. Reste OK pour
  un Postgres sérieux.
- **Calcul à l'ingestion** : 2 inférences BGE-M3 par JP (chaque texte ≤ 8k
  tokens → 1 forward pass chacun), donc ~2× le coût d'embedding sans
  fusion. Négligeable comparé au LLM Step 1.
- **Bénéfice attendu** : grosse amélioration sur les requêtes où l'avocat
  veut explicitement un précédent factuel OU un fondement juridique
  identique (cas dominant en pratique).

#### Combinaison avec stratégies B et C

Compatible avec C : on peut avoir `embedding_situation` et
`embedding_juridique` **par argument** (4 colonnes vectorielles sur
`jp_argument`), ce qui donne la résolution la plus fine possible. Sans
doute overkill pour démarrer — commencer par hybride sur `jp_step1` (B-like),
mesurer le gain, étendre à `jp_argument` si payant.

#### DDL pgvector pour cette approche

```sql
ALTER TABLE jp_step1
  ADD COLUMN embedding_situation vector(1024),
  ADD COLUMN embedding_juridique vector(1024);

CREATE INDEX jp_step1_ann_situation_idx ON jp_step1
  USING ivfflat (embedding_situation vector_cosine_ops) WITH (lists = 200);
CREATE INDEX jp_step1_ann_juridique_idx ON jp_step1
  USING ivfflat (embedding_juridique vector_cosine_ops) WITH (lists = 200);

-- Optionnel : même chose au niveau argument (stratégie C × hybride)
-- ALTER TABLE jp_argument
--   ADD COLUMN embedding_situation vector(1024),
--   ADD COLUMN embedding_juridique vector(1024);
```

### 7.7 Modèle d'embedding recommandé

Cf. journal 2026-05-05 §7.1 et `Note-Optimisation-Embedding-Completion` :
priorité **BGE-M3** (long contexte 8k, multilingue SOTA, dense+sparse+colbert),
à benchmarker contre `multilingual-e5-large` et
`dangvantuan/sentence-camembert-large` sur 4 critères :
- Rang médian des GT (gold truth) sur les 8 questions CRFPA pénales
- Recall top-10 / top-100 / top-1 000
- Coût stockage (1024d vs 768d × N vecteurs)
- Latence d'inférence

À faire sur cluster en **Phase B** une fois ce pipeline Step 1 terminé.

---

## 8. Prochaines étapes (ordre)

| # | Action | Bloquant pour |
|---|---|---|
| 1 | **Re-pilote post-fix `response_format`** sur cluster | Tout le reste |
| 2 | `bash inspect_pilot.sh` — valider qualité juridique FR + distribution `tokens_out` | Décision max_tokens |
| 3 | **Tuner `max_tokens`** (probable cut 4000 → 1500–2000) | Économies massives sur le run complet |
| 4 | **Sharding multi-GPU data-parallel** : ajouter `--shard-idx K --shard-count N` à `run_step1.py`, runner SLURM templaté | Run complet faisable en temps raisonnable |
| 5 | **Run complet** 1,12 M JP en N×L40S parallèles (~quelques jours) | Tout le downstream |
| 6 | **Ingestion DB** : écrire `ingest_jsonl_to_db.py` qui consume `outputs/step1_shard_*/` et alimente les 4 tables, avec normalisateur articles | Recherche utilisable |
| 7 | **Phase B embedding** : embedder les champs §7.1 avec BGE-M3, indexer (pgvector ou FAISS) | Recherche par similarité |
| 8 | **Extension juridictions** : enrichir corpus CE/CAA/TA depuis Légifrance, étendre taxonomie v2 (admin + européen), router via mêmes 3 préambules (CJUE/CEDH → 4e à créer) | Périmètre cible final |

---

## 9. Décisions en attente

- **Qualité juridique de gemma-4-31B en FR** : à juger sur les premiers
  records `ok` du re-pilote (notamment verbatim `attendu_cle`, calibrage
  `synthese_pour_avocat`, fidélité des `cited_articles`).
- **Budget GPU multi-jobs** : combien de L40S en parallèle, inclure H100 ?
  Affecte la durée du run complet (de ~65 j sur 1 GPU à ~2–3 j sur 8–12 GPU).
- **DB engine final** : Postgres (recommandé, supporte pgvector pour
  étape 7) vs DuckDB (analytique, mais moins bon pour transactionnel +
  embedding).
- **Normalisateur articles** : adapter `iterate_regex.py` (regex v5
  local) ou écrire un parser dédié ; gérer les codes peu fréquents
  (CESEDA, code des transports, code du sport…).
- **Politique des cas `oversized`** : aujourd'hui « droppés » au backlog
  (~1 575 JP, <0,2 %). Plan : passe ultérieure en batch OpenAI/Claude
  (gros contextes). À faire après le run open-source.

---

## 10. Pointeurs essentiels

### 10.1 Code
- Tout est sous `05-Technique/benchmark/jp_analysis/`
- Tests verts (53) : `cd 05-Technique/benchmark/jp_analysis && python -m pytest -q`

### 10.2 Documentation
- **Spec de design** (avec §15 résolutions adversariales) :
  `docs/superpowers/specs/2026-05-19-pipeline-step1-jp-analysis-design.md`
- **Plan d'implémentation TDD** (13 tâches) :
  `docs/superpowers/plans/2026-05-19-jp-analysis-step1.md`
- **Taxonomie thèmes canonique** (18 branches v1.0.0) :
  `docs/superpowers/specs/themes-taxonomy-jp.md`

### 10.3 Journaux clés
- Pivot Étape 1 (méthode + ce qu'on cherche à mesurer côté embedding) :
  `01-Projet/journal/2026-05-05.md`
- Note formelle sur la complétion d'embeddings par optimisation
  (Laplacien) — concerne l'Étape 2 future :
  `01-Projet/presentations/Note-Optimisation-Embedding-Completion.pdf`

### 10.4 Cluster
- `pilot_slurm.sh` : job autonome (pilote + bench concurrence). Soit
  `sbatch pilot_slurm.sh` (header SBATCH partition L40S configuré), soit
  `bash pilot_slurm.sh` depuis un nœud GPU déjà alloué (interactive).
- `cluster_check.sh` : détecte partition/compte/L40S — à relancer si tu
  changes de cluster.
- `inspect_corpus.sh` : détecte format réel du corpus (parquet/JSONL).
- `inspect_pilot.sh` : récap qualité après pilote.
- Allocation interactive habituelle (Matthieu) :
  `sinteractive -p mm -w nodemm06 --mem 45G --time 10:00:00`

### 10.5 Risques connus (mémoires persistantes)
- **Env `--user` partagé fragile** : un `pip install` non pinné peut
  casser le venv partagé sur le cluster. **Solution** : on utilise un venv
  isolé dédié (`~/.venv-jp-analysis`) construit via `uv`, configuré dans
  `pilot_slurm.sh`. Ne JAMAIS `pip install` sans préciser `--user` à un
  endroit non isolé.
- **`max_model_len` doit être fixé explicitement** (cf. spec §4 +
  finding adversarial #4). Le serveur vLLM le vérifie au démarrage et
  abort si incohérence.

---

## 11. Annexe — exemple d'output attendu (1 record `ok`)

Pour fixer les idées sur ce que produit la pipeline et ce qui finit en
base. Cet exemple est hypothétique mais représentatif :

```json
{
  "id": "5fca4e3abd3db21cbdd84d1c",
  "number": "18-12.345",
  "juris": "CC",
  "status": "ok",
  "contexte": "Cour de cassation, chambre commerciale, pourvoi en matière de rupture brutale de relations commerciales établies.",
  "arguments_parties": [
    {"partie": "demandeur au pourvoi (fournisseur)", "argument": "la rupture sans préavis suffisant viole l'art. L. 442-1, II du code de commerce", "reponse_juge": "moyen retenu : la cour d'appel a inversé la charge de la preuve sur la durée du préavis raisonnable"}
  ],
  "fondements_retenus": "article L. 442-1, II du code de commerce ; jurisprudence sur la durée du préavis fonction de l'ancienneté et de la dépendance économique",
  "dispositif": "CASSE ET ANNULE l'arrêt de la Cour d'appel de Paris en ce qu'il a rejeté la demande indemnitaire ; renvoie devant la Cour d'appel de Versailles",
  "attendu_cle": "Vu l'article L. 442-1, II du code de commerce ; attendu que la rupture, même partielle, d'une relation commerciale établie sans préavis tenant compte notamment de la durée de la relation et de la dépendance économique du partenaire, engage la responsabilité de son auteur ; qu'il appartient à la juridiction d'apprécier souverainement la durée du préavis raisonnable…",
  "cited_articles": ["L. 442-1 code de commerce", "article 1240 code civil"],
  "solution_resume": "Cassation pour inversion de la charge de la preuve sur la durée du préavis raisonnable en cas de rupture brutale.",
  "dispositif_summary": "Arrêt d'appel cassé et renvoyé devant Versailles.",
  "synthese_pour_avocat": "Un fournisseur poursuit son distributeur pour rupture brutale de relation commerciale établie sans préavis. La Cass. com. casse l'arrêt sur le fondement de l'art. L. 442-1, II C.com., en jugeant que la juridiction doit apprécier la durée du préavis raisonnable au regard de la durée de la relation et de la dépendance économique ; la charge de la preuve ne pèse pas sur la partie qui invoque la brutalité.",
  "dispositif_nature": "CASSE ET RENVOIE",
  "themes": [
    {"branche": "Concurrence, distribution et propriété intellectuelle",
     "sous_branche": "rupture brutale de relations commerciales établies"}
  ],
  "themes_valid": true,
  "themes_taxonomy_version": "1.0.0",
  "schema_version": "1.0.0",
  "model": "QuantTrio/gemma-4-31B-it-AWQ",
  "prompt_variant": "cassation",
  "tokens_in": 4823,
  "tokens_out": 1247,
  "duration_ms": 18432,
  "attempt_count": 1,
  "error_class": null,
  "error_message": null
}
```

---

**Bon courage. La doc spec/plan + les tests TDD couvrent normalement tout
ce qui peut casser ; les commits sont décrits proprement (git log
--oneline -- 05-Technique/benchmark/jp_analysis) ; les helpers shell
(`cluster_check.sh`, `inspect_corpus.sh`, `inspect_pilot.sh`) sont là pour
réduire le temps d'aller-retour avec le cluster.**

— Matthieu, 2026-05-22
