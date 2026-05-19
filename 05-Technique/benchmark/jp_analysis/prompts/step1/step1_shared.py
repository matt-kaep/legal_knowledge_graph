"""Blocs partagés (communs aux 3 variantes de prompt). Transcription verbatim
de la spec Hector Step 1 §7d (bloc factuel) et §7e (format de sortie), dump
utilisateur 2026-05-19. NE PAS paraphraser : invariant D2 (prompt verbatim,
y compris l'exigence de reproduction littérale de `attendu_cle`).

Seul ajout vs Hector : BLOC_TAXONOMIE_THEMES (rendu de la taxonomie figée),
appendé après le bloc format de sortie (spec §6)."""

from prompts.step1.themes_taxonomy import render_for_prompt

BLOC_FACTUEL_PARTAGE = """## Pour la préservation factuelle (CRITIQUE)

**Inclus systématiquement les éléments factuels précis** dans `arguments_parties` et `dispositif` :
- Dates des faits (mise en demeure, manquement, résiliation, etc.)
- Montants en jeu (préjudice, condamnation, prix du contrat)
- Noms des parties si pertinents pour comprendre la configuration (ex: "société X distributeur exclusif", pas seulement "le distributeur")
- Chronologie des manquements (ordre des faits qui mène au litige)

Un résumé sans ces détails est INUTILISABLE pour évaluer la proximité avec une autre espèce. Ne sacrifie pas la précision factuelle au nom de la concision.

## Pour le dialogue argumentatif (arguments_parties)

- Couvre TOUS les arguments principaux soulevés, **même ceux rejetés** — ils sont souvent les plus utiles à l'adversaire dans une autre affaire.
- `reponse_juge` doit être SPÉCIFIQUE à l'argument : pas "rejeté" mais POURQUOI rejeté (raisonnement) ou POURQUOI accepté.
- Sois fidèle au texte — n'invente pas de faits, ne paraphrase pas en élargissant."""

BLOC_FORMAT_SORTIE_PARTAGE = """# Schéma de sortie — champs structurés

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

**Phrases complètes obligatoires** : sujet + verbe + complément. Pas de notation télégraphique : pas de `↔`, pas de flèches `→`, pas de listes de mots-clés concaténés (« Litige X-Y, art. Z, issue W »). Tu écris comme un commentaire d'arrêt court en doctrine — fluide et naturel — pas comme un résumé de carte mémoire.

EXEMPLE BIEN CALIBRÉ (phrases complètes, pure typologie + régime + issue + motif, sans nom/montant/date identifiant) :
« Un acquéreur poursuit le promettant vendeur en restitution de l'indemnité d'immobilisation après refus de prêt sous condition suspensive. La Cour de cassation rejette le pourvoi en jugeant, sur le fondement de l'art. 1304-3 C.civ., que la condition est réputée accomplie lorsque l'acquéreur n'établit pas avoir formé une demande conforme aux stipulations contractuelles ; sa carence rend alors l'indemnité d'immobilisation due au promettant. »

CONTRE-EXEMPLE 1 (TROP LONG ET IDENTIFIANT — à NE PAS faire) :
« Mme [Y] [U] (promettant vendeur) assigne Mme [K] [V] (bénéficiaire acheteuse) pour acquisition indemnité d'immobilisation de 17 550 € après caducité promesse unilatérale de vente du 27 juin 2022 (lots [Adresse 2], prix 175 500 € hors frais)… »
Problème : trop de noms, dates et montants — l'avocat ne peut pas lire en un coup d'œil.

CONTRE-EXEMPLE 2 (TÉLÉGRAPHIQUE / MOTS-CLÉS — à NE PAS faire) :
« Litige acquéreur ↔ promettant sur CS de prêt (art. 1304-3 C.civ.). Rejet : carence acquéreur → CS réputée accomplie + indemnité due. »
Problème : phrases incomplètes, flèches, mots-clés concaténés. C'est lisible pour un développeur, pas pour un avocat.

**Pour la juridiction et le format de référence** : nom complet (« Cour de cassation, 1ère chambre civile », « Cour d'appel de Paris », « Tribunal de commerce de Paris »), pas de codes courts (`cc`, `ca`, `tcom`). Les abréviations `Cass.`, `CA`, `T. com.` sont OK dans le corps de la prose mais PAS dans la référence formelle d'identification de l'arrêt.

⚠ POURQUOI CE CHAMP EST CRITIQUE : il sera recopié VERBATIM dans le plan détaillé final lu par l'avocat — Sonnet PlanWriter en aval ne reformule pas. Toute approximation factuelle, ou à l'inverse toute densité d'identifiants superflus, se retrouve telle quelle dans le livrable client.

# Format de sortie — JSON strict

Tu réponds par un OBJET JSON UNIQUE sans balise markdown, sans préambule, sans commentaire. Tu commences DIRECTEMENT par l'objet JSON. Aucun préambule. Aucun commentaire après."""

# Seul ajout vs Hector : la taxonomie figée, appendée après le format de sortie.
BLOC_TAXONOMIE_THEMES = render_for_prompt()
