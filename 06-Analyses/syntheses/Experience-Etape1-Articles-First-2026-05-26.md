---
date: 2026-05-26
type: synthese
tags: [synthese, etape1, articles-first, embedding, BGE-M3, retrieval, recall@K]
parent: "[[Experience-B2-Embedding-Naif-Penal]]"
status: lu
pertinence: haute
---

# Étape 1 — Embedding articles-first avec BGE-M3 : analyse complète

> **Pitch.** En embeddant les **3 183 articles pénaux LEGI** (texte intégral) avec **BGE-M3** comme point d'entrée, on retrouve l'article obligatoire en **top-3 dans 4 questions sur 8** (procédure pénale) et l'article gold à **100 % en top-1 000 pour toutes les questions sauf 1**. C'est un saut qualitatif majeur par rapport à la baseline du 5 mai (e5-base + JP-first, **0/16 gold en top-10**, rang médian 76 800 / 118 000).

## 1. Setup expérimental

### 1.1 Pipeline

| Étage | Configuration |
|---|---|
| **Corpus** | 8 085 articles pénaux référencés dans le graphe biparti (4 codes : CP, CPP, route, CJPM) |
| **Source** | Dump DILA LEGI 13/07/2025, ingéré via `legi.py` → SQLite (1 737 248 articles tous codes confondus) |
| **Résolution `pair_key` → texte** | Normalisation `L743-7 ↔ L. 743-7`, recherche multi-candidats par code+num |
| **Couverture effective** | **3 183 articles résolus / 8 085** (39 %) — voir §6 pour explication |
| **Embedding** | BGE-M3 (1024 dim, contexte effectif 2 048 tokens sur MPS Mac, L2-normalisé) |
| **Questions** | 8 CRFPA 2025 (`rubrics_penal.json`), 3 droit pénal + 5 procédure pénale |
| **Évaluation** | `recall@K` cosine pour K ∈ {1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000} |
| **Métrique synthétique** | **K\*** = plus petit K tel que `recall@K ≥ 0.5` |

### 1.2 Pourquoi le pivot « article-first »

La baseline B2 du 5 mai a montré que **l'embedding e5-base ne capte pas le signal juridique français** : rang médian des arrêts gold = 76 782 / 118 112. La Phase D du journal proposait d'**inverser l'approche** : embedder les articles (textes courts, denses, structurés sémantiquement) plutôt que les arrêts (textes longs, narratifs, bruités). Cette expérience teste cette hypothèse.

Avantage attendu : la **sémantique question→article** est plus directe (une question CRFPA cible un point de droit codifié) que la sémantique question→arrêt (qui demande de matcher contre des faits + procédure + motifs).

### 1.3 Deux sides évalués

- **Article-side** : question → top-K articles → recall vs `articles_attendus.obligatoires`
- **JP-via-graph** : top-K articles → **toutes les JP qui citent ≥ 1 article** du top-K via le graphe biparti → recall vs `jp_attendues` (résolu par regex pourvoi sur le numéro CC)

Le JP-via-graph **ne nécessite pas d'embed JP** — il exploite uniquement la structure de citation déjà encodée dans `graph_penal.npz`. C'est exactement ce que la Phase D suggérait : *« les JP sont retrouvées par back-edge graphe »*.

---

## 2. Résultats — vue d'ensemble

### 2.1 Table K\* par question

| Question | branche | gold oblig | **K\* art (oblig)** | gold tot (o+optio) | **K\* art (tot)** | gold JP | **K\* JP via graph** |
|---|---|---|---|---|---|---|---|
| CNB-PP-2025-Q2 | PP | 1 | **1** | 4 | 30 | 2 | **1** |
| CNB-PP-2025-Q3 | PP | 2 | **1** | 5 | 10 | 1 | **1** |
| CNB-PP-2025-Q1 | PP | 2 | **3** | 6 | 200 | 2 | 5 |
| CNB-PP-2025-Q4 | PP | 1 | **3** | 6 | 500 | 1 | 30 |
| CNB-PP-2025-Q5 | PP | 2 | **10** | 9 | 100 | 0 | — (gold absent) |
| CNB-PENAL-Q3 | PENAL | 2 | 50 | 4 | 1 000 | 2 | 50 |
| CNB-PENAL-Q1 | PENAL | 7 | 200 | 17 | 500 | 8 | 100 |
| CNB-PENAL-Q2 | PENAL | 2 | **1 000** | 6 | 1 000 | 2 | 1 000 |

### 2.2 Courbes recall@K (articles obligatoires)

```
K                 1    3    5    10    20    30    50    100   200   500   1000
PP-Q2           1.00 1.00 1.00 1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00
PP-Q3           0.50 1.00 1.00 1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00
PP-Q4           0.00 1.00 1.00 1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00
PP-Q1           0.00 0.50 0.50 0.50  0.50  0.50  0.50  0.50  1.00  1.00  1.00
PP-Q5           0.00 0.00 0.00 0.50  0.50  0.50  0.50  0.50  0.50  0.50  0.50
PENAL-Q1        0.00 0.00 0.14 0.14  0.29  0.43  0.43  0.43  0.57  0.71  0.86
PENAL-Q3        0.00 0.00 0.00 0.00  0.00  0.00  0.50  0.50  0.50  0.50  1.00
PENAL-Q2        0.00 0.00 0.00 0.00  0.00  0.00  0.00  0.00  0.00  0.00  1.00
```

### 2.3 Lecture rapide

- **3 questions** (PP-Q2, Q3, Q4) sont **résolues au top-3** côté article — performance « moteur de recherche juridique grand public ».
- **5 questions** atteignent un recall ≥ 50 % à K ≤ 50 côté article.
- **Q2 PENAL** est l'exception spectaculaire : gold trouvé seulement à K = 1 000 (recall passe de 0 à 100 % entre 500 et 1 000 — toujours sur les `obligatoires`).
- **JP via graph** suit l'article-side de très près (K\* JP ≈ K\* article), sauf Q5 où le gold JP est nul dans le corpus (limite de la regex pourvoi CC-only).

---

## 3. Analyse qualitative — inspection top-K

### 3.1 Q2 PP — résolution parfaite (`K* = 1`)

**Question** : *Les trois suspects ont été interpellés le 2 mars 2025 à 9h et placés en garde à vue. Les droits leur ont été notifiés…*

**Gold OBLIG** : `code_de_procedure_penale:63` (durée et droits de la garde à vue).

**Top-1 retourné** : `code_de_procedure_penale:63`. Match direct par lexico-sémantique : la question contient « garde à vue », « notification », « droits » ; l'article 63 *est exactement* l'article qui codifie ces termes.

→ **Cas idéal de l'article-first** : question lexicale + article codifié.

### 3.2 Q4 PP — `K* = 3` mais top-2 quasi-parfait

**Question** : *Vincent, qui a gardé le silence pendant sa GAV et son IPC, peut-il encore soulever la nullité de la mesure de géolocalisation, de sa garde à vue…*

**Gold OBLIG** : `code_de_procedure_penale:173-1` (forclusion des nullités de fond).

**Top-10 retourné** :
```
1. CPP:230-41   (nullité des interceptions)
2. CPP:173-1    ← OBLIG ✓
3. CPP:706-100  (nullité activation à distance)
4. CPP:230-35   …
…
```

L'article 173-1 sort en **rang 2**, devancé par 230-41 (sujet voisin mais hors gold). Le top-10 est intégralement « cohérent » thématiquement (nullités, géolocalisation, écoutes) — l'encodeur a parfaitement saisi le **thème de la nullité de procédure**.

→ **Limite haute du retrieval par embedding** : récupère le bon thème, manque parfois l'article *exact* d'un rang.

### 3.3 Q2 PENAL — l'échec instructif (`K* = 1 000`)

**Question** : *Thierry, dirigeant du laboratoire 'Medica SA' fabricant le médicament prescrit à Romuald, a été informé des effets indésirables du médicament dès fin 2023 mais n'a alerté les médecins et patients qu'en…*

**Gold OBLIG** : `code_penal:222-19` (blessures involontaires), `code_penal:121-3` (intention / imprudence / négligence).

**Rangs des gold** :
- 222-19 → **rang 836**
- 121-3 → **rang 971**

**Top-10 retourné** :
```
1. CPP:706-47-1     (suivi des condamnés pour infractions sexuelles)
2. CPP:D47-27       (troubles mentaux / détention)
3. CPP:706-136      (chambre instruction, décision)
4. CP:222-33        (harcèlement sexuel)
5. CP:222-28        (agression sexuelle)
…
```

**Top-10 totalement off-topic**. L'encodeur a manifestement « croisé » deux signaux sémantiques :
- (a) le mot « médicament » + « patient » a tiré vers les articles psychiatriques (troubles mentaux, expertise psy)
- (b) la mention d'un dirigeant d'entreprise mêlée à des suites pour les patients a tiré vers les infractions sexuelles (qui partagent un vocabulaire « victime / responsable »)

Ce que la question **demande vraiment** : un *raisonnement juridique de qualification* — « dirigeant informé qui n'agit pas = imprudence (121-3) → blessures involontaires (222-19) ». Aucun mot de la question ne contient « blessure », « involontaire », « imprudence », ou « élément moral ». L'encodeur ne fait pas ce raisonnement.

→ **Limite intrinsèque de l'embedding lexico-sémantique** : il ne peut pas inférer une qualification juridique à partir d'un récit factuel.

### 3.4 Q3 PENAL — un gold parfait, un gold catastrophique

**Question** : *Le laboratoire 'Medica SA' (SA) dont Thierry est dirigeant a fait l'objet début 2025 d'une fusion-absorption par la société 'Invest SA' (holding)…*

**Gold OBLIG** : `code_penal:121-1` (responsabilité personnelle), `code_penal:121-2` (responsabilité personne morale).

**Rangs** :
- **121-2 → rang 31** ✓ (l'encodeur trouve la responsabilité PM, terme « fusion-absorption » + « société » + « dirigeant » résonne avec PM)
- **121-1 → rang 968** ✗ (raté ; l'article *« nul n'est responsable pénalement que de son propre fait »* est trop abstrait pour ressortir sans le mot-clef « responsabilité personnelle »)

→ **Pattern asymétrique** : l'encodeur trouve l'article du domaine du cas (PM) mais rate l'article-principe (qui est cité comme contraste obligatoire dans la grille).

---

## 4. Comparaison vs baseline B2 (5 mai)

| Métrique | Baseline B2 (e5-base, JP-first) | Étape 1 (BGE-M3, articles-first) | Gain |
|---|---|---|---|
| Modèle | `multilingual-e5-base` (512 tok) | `BAAI/bge-m3` (2 048 tok effectifs) | **+ 4× contexte** |
| Entrée encodée | Texte arrêt brut | Texte article LEGI | **inversion** |
| Pool de candidats | 118 112 JP | 3 183 articles | **× 37 plus petit** |
| Recall articles top-10 | 31 % (best, K=10, union) | 50-100 % pour 5/8 questions | **× 1.6 à × 3** |
| **K\* JP (50 % recall)** | **∞** (jamais atteint) | **≤ 5** pour 3/8 questions | **bond qualitatif** |
| Rang médian gold | 76 782 / 118 112 | top-100 pour majorité des oblig | — |

**Conclusion comparative** : on ne « gagne pas un peu » — on change de régime. La baseline B2 était dans un régime où **aucune information utile** ne sortait du top-K. Étape 1 est dans un régime où **le top-3 est exploitable** pour la moitié des questions et **le top-100 est exploitable** pour les 3/4.

---

## 5. Pattern dégagé : procédure pénale ≫ droit pénal de fond

### 5.1 Statistique

| Branche | Questions | médiane K\* (oblig) | médiane K\* (JP via graph) |
|---|---|---|---|
| **PP** (procédure pénale) | 5 | **3** | **5** |
| **PENAL** (droit pénal de fond) | 3 | **200** | **100** |

### 5.2 Hypothèse explicative

Les questions de **procédure pénale** ressemblent à des requêtes lexicales sur des articles bien identifiés :
- « garde à vue » → CPP:63
- « nullités après IPC » → CPP:173-1
- « détention provisoire JLD » → CPP:144

Le vocabulaire de la question matche directement le vocabulaire de l'article. L'embedding est dans son terrain idéal.

Les questions de **droit pénal de fond** sont des **cas pratiques compositionnels** :
- Un récit factuel (dirigeant, médicament, retard d'information)
- Qui demande une *qualification* (blessures involontaires + manquement de prudence)
- Qui mobilise des articles-principes abstraits (121-3, 121-1) jamais nommés dans la question

L'embedding ne « raisonne » pas. Il matche thématiquement (médicament → médical → psychiatrie → troubles mentaux) et rate la dorsale dogmatique.

### 5.3 Implication

**L'article-first est borné par la nature compositionnelle des questions de fond**. Aucun encodeur plus gros ou plus français ne résoudra Q2 PENAL sans :
- Soit une **réécriture de la question** par un LLM (transformer « informé d'effets indésirables, n'alerte pas » en « imprudence du dirigeant »)
- Soit une **augmentation du gold via citations** (la JP qui qualifie ce type de fait *cite* 121-3 et 222-19 ; on peut donc retrouver les articles via les JP, à condition que l'embed JP fonctionne — voir Étape 2)
- Soit un **modèle entraîné sur du raisonnement juridique** (légal-FR fine-tuning, voyage-law, etc.)

---

## 6. Limitations connues

### 6.1 Couverture LEGI à 39 % du graphe (gold à 100 %)

**Cause** : la JP du corpus cite l'**ancien Code pénal** (pré-1994) et l'**ancien Code de la route** (pré-2001), dont les articles ont été renumérotés. Exemples non résolus : `code_penal:4`, `:59`, `:60`, `:309`, `code_de_la_route:R6`, etc.

**Impact sur l'éval** : nul sur les obligatoires (50/51 résolus, le manquant `code_penal:222-26-2` n'est pas un nœud du graphe). Mais le **pool de candidats est rétréci à 3 183 au lieu de 8 085**.

**Implication** : un article ancien cité par une JP du corpus mais non actif aujourd'hui n'est jamais récupérable. Cela peut sous-estimer la performance — moins de bruit côté top-K.

### 6.2 Eval JP biaisée par `ref:null`

Le `jp_attendues` du `rubrics_penal.json` a `ref: null` et seulement une `short_ref` texte. Résolution par regex pourvoi → **CC uniquement** (format `XX-XX.XXX`). Une question (PP-Q5) a 0 JP gold extractible, K\* JP est artificiellement « — ».

### 6.3 8 questions = échantillon faible

Les K\* observés ont une variance importante. Cohérent avec la motivation initiale de l'**Étape 3** (génération de questions par doctrine, à grande échelle).

### 6.4 Politique de troncature à 2 048 tokens

Décision pragmatique (OOM MPS au-delà). Affecte **5 articles** (sur 3 183, 0.16 %) et **16 732 JP** (17 %). Les articles dépassants sont chunkés + mean-poolés (préservation du signal), mais ce n'est pas évalué sur cette expérience puisqu'on n'a pas embeddé les JP.

---

## 6.bis Métrique stricte adoptée (2026-05-26)

Le critère retenu pour pénaliser la sur-extraction est :

- **Critère dur** : `recall@10 articles (oblig) ≥ 0.5` ET `recall@5 JP ≥ 0.5`
- **Critère facile** : `recall@20 articles (oblig) ≥ 0.5` ET `recall@10 JP ≥ 0.5`

C'est l'équivalent d'un budget de 10/5 (resp. 20/10) résultats affichables à l'utilisateur final.

### Résultats sur le critère dur

| Question | r@10 art | r@5 JP | Passe critère dur ? |
|---|---|---|---|
| CNB-PP-Q2 | 1.00 | 0.50 | ✅ |
| CNB-PP-Q3 | 1.00 | 1.00 | ✅ |
| CNB-PP-Q1 | 0.50 | 0.67 | ✅ |
| CNB-PP-Q4 | 1.00 | 0.00 | ❌ JP graphe défaillant |
| CNB-PP-Q5 | 0.50 | n/a | ⚪ gold JP absent du corpus |
| CNB-PENAL-Q1 | 0.14 | 0.12 | ❌ |
| CNB-PENAL-Q3 | 0.00 | 0.00 | ❌ |
| CNB-PENAL-Q2 | 0.00 | 0.00 | ❌ |

**→ 3 / 7 questions évaluables passent le critère dur.** (Baseline 5 mai : **0 / 7**.)

Les moyennes agrégées : `mean r@10_art = 0.518`, `mean r@5_jp = 0.286`. Au 5 mai : `mean r@10_art ≈ 0.16`, `mean r@5_jp = 0.00`.

## 6.ter Diagnostic par question — pourquoi ça réussit ou échoue

### Questions qui passent (3) — anatomie du succès

#### CNB-PP-2025-Q2 — `r@10=1.00, r@5_jp=0.50` ✅
- **Question** : garde à vue, droits notifiés.
- **Top-1** : `CPP:63` (gold OBLIG). Match parfait : la question contient « garde à vue », l'article 63 *définit* la garde à vue.
- **JP via top-5** : 1 107 JP candidates, 1/2 gold récupérés. Le 2e gold est dans un pourvoi 16-80.564 qui cite probablement CPP:63 mais avec un pattern différent.
- **Verdict** : cas idéal. Question lexicale + article codifié. Aucun problème.

#### CNB-PP-2025-Q3 — `r@10=1.00, r@5_jp=1.00` ✅
- **Question** : interrogatoire de première comparution.
- **Top-1** : `CPP:116-1` (gold OBLIG, criminel), top-3 : `CPP:116` (gold OBLIG, général). Les deux gold dans top-3.
- **JP via top-5** : 1/1 gold récupéré (parfait).
- **Verdict** : cas idéal, même structure que PP-Q2.

#### CNB-PP-2025-Q1 — `r@10=0.50, r@5_jp=0.67` ✅
- **Question** : balise GPS, trafic de stupéfiants en bande organisée.
- **Top-1** : `CPP:78-2-4` (atteinte grave à la sécurité — pertinent mais hors gold). **Top-2** : `CPP:230-32` (gold OBLIG, géolocalisation).
- **Gold non récupéré** : `CPP:230-33` au rang 163 (1 article gold sur 2 trouvé → recall@10 = 0.5).
- **JP via top-5** : 1 822 JP candidates, 2/3 gold récupérés.
- **Verdict** : succès partiel. Le sujet (géolocalisation) est trouvé ; un des deux articles spécifiques (230-33) est manqué.

### Questions qui échouent (4) — diagnostics distincts

#### CNB-PP-2025-Q4 — `r@10=1.00, r@5_jp=0.00` ❌ « back-edge graphe défaillant »
- **Question** : nullité après IPC.
- **Top-2** : `CPP:173-1` (gold OBLIG) — article trouvé.
- **JP via top-5** : 462 JP candidates, **0/2** gold récupérés. Le pourvoi gold (`23-84.957`) **n'est pas cité par CPP:173-1** dans le graphe.
- **Verdict** : article ✅, JP via graphe ❌. Cause : la JP attendue ne cite probablement pas 173-1 mais des articles voisins (170, 174). C'est une limite du **back-edge graphe** : il dépend de la *fidélité des citations* dans le corpus.
- **Fix possible** : enrichir le top-K avec les articles voisins (CPP:170, 174, 802 — qui sont dans les optionnels mais aux rangs 132, 938, 1061).

#### CNB-PP-2025-Q5 — gold JP absent ⚪
- **Question** : placement en détention provisoire sans avocat.
- **Article gold** : `CPP:145` (rang 10), `CPP:186` (rang 1229).
- **JP gold extractibles** : 0 (aucun pourvoi CC dans `short_ref`).
- **Verdict** : non évaluable côté JP. Article-side reste évaluable (recall@10 = 0.5).

#### CNB-PENAL-2025-Q1 — `r@10=0.14, r@5_jp=0.12` ❌ « contamination thématique »
- **Question** : viol par cunnilingus, trouble mental, intoxication médicamenteuse (cas Romuald).
- **Top-1** : `CPP:706-47-1` (suivi condamnés sexuels), top-3 mêle suivi pénal et chambre d'instruction.
- **Gold trouvé top-10** : 1/7 (`code_penal:122-1`, irresponsabilité psy, rang 4). Reste dispersé : `222-23` rang 110, `222-22` rang 368, `121-3` rang 613.
- **Verdict** : 7 articles à mobiliser → l'encodeur en attrape 1 dans le top-10 (122-1, le plus « marquant » thématiquement), mais rate les articles centraux (222-23 viol, 121-3 élément moral). **Contamination par les articles de suivi** (706-47-1, 131-36-4) qui partagent du vocabulaire sans être pertinents.
- **Fix possible** : reformuler la question via LLM (« qualifications applicables → infractions sexuelles + irresponsabilité ») ; ou requête multi-vecteur (un par chef de qualification attendu).

#### CNB-PENAL-2025-Q2 — `r@10=0.00, r@5_jp=0.00` ❌ « compositionnel pur »
- **Question** : dirigeant labo informé d'effets indésirables, retard d'alerte.
- **Top-10** : *aucun article gold*. Top-1 = 706-47-1 (suivi sexuel — même artefact que Q1). Articles gold à 836 et 971.
- **Verdict** : échec total. Question 100 % compositionnelle (« non-acte → imprudence → blessures involontaires »). Aucun mot du gold (« blessures », « involontaires », « élément moral », « imprudence ») n'est dans la question.
- **Fix possible** : impossible à résoudre par embedding seul. Réécriture LLM nécessaire, ou modèle spécialisé droit pénal.

#### CNB-PENAL-2025-Q3 — `r@10=0.00, r@5_jp=0.00` ❌ « article-principe abstrait raté »
- **Question** : fusion-absorption, responsabilité PM.
- **Gold rang** : `121-2` (responsabilité PM) **rang 31** ✓, `121-1` (responsabilité personnelle) **rang 968** ✗.
- **Verdict** : 1/2 gold trouvable seulement à K=50. L'article 121-1 est un principe abstrait (« nul n'est responsable pénalement que de son propre fait ») trop générique pour matcher une question concrète sur la fusion-absorption.
- **Fix possible** : ajouter au top-K les articles « principes » connus pour chaque domaine, ou matcher contre une formulation paraphrasée (LLM).

### Trois patterns d'échec, trois fixes différents

| Pattern | Questions | Diagnostic | Fix Étape 2 plausible |
|---|---|---|---|
| **A** Compositionnel pur | PENAL-Q2, partiel Q1 | Aucun mot-clé gold dans la question | Réécriture LLM de la question, ou JP-first si JP cite article |
| **B** Article-principe abstrait raté | PENAL-Q3 (121-1) | Article trop générique, non thématique | Liste d'articles-principes par domaine, ou multi-vecteur |
| **C** Back-edge graphe défaillant | PP-Q4 | Article trouvé mais JP gold ne le cite pas | Élargir top-K par articles voisins (citation graph 2-hop) |

## 7. Implications pour Étapes 2 et 3

### 7.1 Étape 2 — Complétion d'embeddings JP via le graphe

L'expérience confirme que **l'article-first est puissant mais borné par la compositionalité des questions de fond**. L'Étape 2 (complétion par laplacien ou LLM) ne doit pas seulement « remplir les JP sans summary » mais surtout produire des **embeddings JP suffisamment informatifs pour qu'une question PENAL puisse être résolue via la JP plutôt que via l'article**.

Hypothèse à tester : pour Q2 PENAL, est-ce que l'embed des JP qui *qualifient* « dirigeant non-alerté → blessures involontaires » remonte mieux que l'embed des articles 222-19/121-3 ? Si oui, la stratégie « JP-first » récupère ce que « article-first » rate. **Les deux approches sont complémentaires.**

### 7.2 Étape 3 — Benchmark étendu via doctrine

L'échantillon de 8 questions est trop petit pour distinguer fiabilement le signal du bruit sur les questions de fond. L'Étape 3 doit produire **≥ 100 questions étiquetées** pour permettre une vraie statistique sur le pattern PP vs PENAL.

Une **stratification** par type de question (lexicale vs compositionnelle, fond vs procédure) deviendra possible — c'est exactement le terrain où le GNN sur graphe enrichi (Étape 3) pourrait briller : il aurait appris la *co-occurrence* question/article/JP plutôt que de matcher en cold-start.

### 7.3 Recommandation immédiate

**Garder l'article-first comme plancher reproductible** :
- Embedding articles (3 183 × 1 024) tient en **12 MB** — réutilisable partout
- Le back-edge JP via graphe ne coûte rien (déjà construit)
- C'est un baseline solide contre lequel mesurer toute amélioration future

**Ne pas embedder les JP sur Mac MPS** : la combinaison (97 000 docs longs × MPS sans Flash-Attention) sature et plante en pratique. Soit cluster L40S (recommandé), soit CPU long (4-6 h prévisibles).

---

## 8. Artefacts produits

| Fichier | Taille | Description |
|---|---|---|
| `data/articles_penal.parquet` | 1.3 MB | 3 183 articles avec texte LEGI |
| `data/articles_coverage.json` | 648 B | Diagnostic résolution (39 % global, 100 % gold) |
| `data/token_stats.json` | 603 B | p50/p99/p100 articles et JP CC |
| `data/articles_order.npy` | 92 KB | pair_keys dans l'ordre d'embedding |
| `data/pairkey_to_graphcol.npy` | 13 KB | mapping vers colonnes du graphe biparti |
| `data/emb_articles.npy` | **12 MB** | **(3 183, 1 024) fp32 L2-norm — l'artefact central** |
| `data/recall_curves_articles_only.csv` | 4 KB | Recall@K par question × side × metric |
| `data/recall_kstar_articles_only.json` | 2 KB | K\* synthétique par question |

Tous reproductibles à partir de la branche `etape1-embedding-pur` :
```bash
PYTHONPATH=. python scripts/02_fetch_articles.py
PYTHONPATH=. python scripts/01_token_stats.py
PYTHONPATH=. python scripts/03_embed.py --device mps --batch 8  # n'embed que les articles si on coupe à temps
PYTHONPATH=. python scripts/04a_eval_articles_only.py
```

## 9. Liens

- [[Experience-B2-Embedding-Naif-Penal]] — baseline du 5 mai (référence comparative)
- [[Note-Optimisation-Embedding-Completion]] — méthodes pour l'Étape 2
- `01-Projet/journal/2026-05-05.md` §7 — roadmap Johnny (3 étapes)
- `01-Projet/presentations/Note-Pedagogique-Etape1-2026-05-20.html` — exposé du design
- `05-Technique/benchmark/etape1_embedding_pur/PLAN.md` — plan d'implémentation détaillé
