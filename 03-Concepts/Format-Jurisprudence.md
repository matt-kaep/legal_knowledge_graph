---
tags: [concept, format, jurisprudence, normalisation, benchmark, kg]
type: format-canonique
aliases: [Format-JP, Format-Decision, Normalisation-Jurisprudence]
domaine: transverse
status: proposition
created: 2026-04-20
modified: 2026-04-20
---

# Format canonique de la jurisprudence

> Référence unique pour représenter une décision ou une référence à une décision, dans le KG, le benchmark et la ground truth.

Complète [[Format-Fondement-Juridique]] (pendant pour les codes et articles) et étend [[schema.py]] `JPReference`.

---

## 1. Identifiant canonique

### Priorité des identifiants

1. **ECLI** (European Case Law Identifier) — standard européen, prioritaire
   - Format : `ECLI:FR:CCASS:2018:17.21.456` ou `ECLI:FR:CA:PARIS:2022:...`
2. **ID Judilibre** — fallback quand ECLI absent (ex : `6079411b9ba5988459c4064e`)
3. **ID synthétique** — pour cas inventés du benchmark (M6), format : `SYNTH-<uuid>`

### Règle d'unicité

Dans le KG, **un nœud `Decision` = un ECLI**. Si une décision a plusieurs IDs (Judilibre + ECLI), garder les deux comme attributs mais l'ECLI fait foi.

---

## 2. Référence courte (format humain)

Format standardisé pour affichage, citation dans rubriques, logs :

```
{JURIDICTION} {CHAMBRE}, {DATE}, n° {NUMERO}
```

| Décision | Référence courte |
|---|---|
| Arrêt Cass. 2e civ. 19/05/1998 n° 96-20.991 | `Cass. civ. 2e, 19 mai 1998, n° 96-20.991` |
| CA Paris pôle 5 ch. 4 du 14/03/2022 | `CA Paris, 14 mars 2022, n° 20/12345` |
| TJ Lyon 10/01/2024 | `TJ Lyon, 10 janv. 2024, n° 22/00321` |

### Normalisation (abréviations juridiques FR)

- `Cass.` (Cour de cassation), `CA` (Cour d'appel), `TJ` (Tribunal judiciaire), `TC` (Tribunal de commerce), `CE` (Conseil d'État), `CConst` (Conseil constitutionnel)
- Chambres : `civ. 1re`, `civ. 2e`, `civ. 3e`, `com.`, `soc.`, `crim.`, `ass. plén.`, `ch. mixte`
- Dates : format long français `19 mai 1998` (pas `19/05/1998` en référence de citation)

---

## 3. Schéma complet d'une décision (nœud du KG)

Étend `Decision` de `schema.py:106` :

```python
class Decision(BaseModel):
    # Identifiants
    id: str                        # ECLI prioritaire, sinon Judilibre
    ecli: str | None = None
    judilibre_id: str | None = None

    # Métadonnées
    juridiction: Juridiction       # enum : CC, CA, TJ, TC, CE, ...
    chambre: str | None = None     # "civ. 2e", "com.", "soc." ...
    date: date
    numero: str | None = None      # "96-20.991", "20/12345"
    formation: str | None = None   # "formation plénière", "section"

    # Contenu
    full_text: str
    structure: DecisionStructure | None = None   # faits/moyens/motifs/dispositif
    zones: dict | None = None                     # offsets Judilibre

    # Sens et portée
    sens: SensArret                # cassation / rejet / confirmation / ...
    publication: PublicationLevel | None = None   # P/B/R/L - niveau de publication CC
    solution_summary: str | None = None           # résumé 2-4 phrases

    # Fondements cités
    visa_articles: list[str] = []          # clés canoniques code:article (cf. Format-Fondement-Juridique)
    cited_decisions: list[str] = []        # IDs des décisions citées
    cites_revirement_of: str | None = None # ID de la décision revirée (si applicable)

    # Méta
    is_synthetic: bool = False
```

### Enum `PublicationLevel` (spécifique Cour de cassation)

Indicateur clé de l'**importance jurisprudentielle** — à exploiter pour classer la JP :

| Code | Signification | Importance |
|---|---|---|
| `P` | Bulletin (Publié au Bulletin) | Élevée |
| `B` | Bulletin + diffusion large | Élevée |
| `R` | Rapport annuel | Maximale (arrêt de principe) |
| `L` | Lettres de chambre | Moyenne |
| (vide) | Inédit | Faible |

> L'indicateur de publication est disponible dans Judilibre et sert de **proxy objectif de l'importance** pour le ranking.

---

## 4. Référence à une JP depuis la rubrique / ground truth

Étend `JPReference` (`schema.py:118`) — utilisé dans le benchmark pour lister les JP attendues dans la réponse :

```python
class JPReference(BaseModel):
    # Identité
    decision_id: str               # ECLI ou ID Judilibre
    juridiction: Juridiction
    date: date
    short_ref: str                 # "Cass. civ. 2e, 19 mai 1998, n° 96-20.991"

    # Rôle dans le raisonnement
    rank: JPRank                   # principe | interpretation | espece (cf. schema.py)
    importance: Literal[1, 2, 3]   # 1=principe de référence, 2=confirmation, 3=illustration

    # Position vis-à-vis de la question / du client
    sens_vs_question: Literal["favorable", "defavorable", "nuance", "neutre"]

    # Justification
    why: str                       # pourquoi cette décision est citée ici
    key_paragraphs: list[str] = [] # extraits saillants (optionnel)
```

### Le champ `sens_vs_question` (nouveauté)

Indispensable pour l'**objectif métier** : *"sortir les JP qui vont dans le sens du client et celles qui vont dans le sens adverse"*.

| Valeur | Signification |
|---|---|
| `favorable` | Soutient la position (du demandeur si `client_position=demande`) |
| `defavorable` | Affaiblit la position |
| `nuance` | Position modérée, ni pour ni contre franchement |
| `neutre` | Citée pour contexte/définition, pas pour trancher |

> C'est un champ **propre à la question posée**, pas une propriété intrinsèque de l'arrêt. Le même arrêt peut être `favorable` pour une question et `defavorable` pour une autre.

---

## 5. Classification hiérarchique (`rank`)

Repris de `JPRank` (`schema.py:37`) :

| Rank | Juridiction typique | Rôle |
|---|---|---|
| `principe` | Cour de cassation (surtout si publié R/P) | Pose la règle de droit |
| `interpretation` | Cours d'appel | Précise, interprète, module |
| `espece` | TJ / TC / premières instances | Applique aux faits d'un cas particulier |

Cette taxonomie permet au benchmark de **distinguer deux niveaux de questions** (cf. [[Benchmark-KG-Juridique-FR-Design]] §2) :
- **Niveau 1 (Principe QA)** : attend des `principe` + éventuellement `interpretation`
- **Niveau 2 (Applied QA)** : attend en plus des `espece` factuellement proches

---

## 6. Temporalité

Chaque décision a 3 dates à distinguer :

| Champ | Signification |
|---|---|
| `date` | Date de rendu de la décision |
| `date_applicability_start` | À partir de quand elle fait autorité (souvent = `date`) |
| `date_overruled` / `revirement_par` | Si revirée, ID de l'arrêt de revirement et date |

Pour les requêtes temporelles (Module 5 du benchmark) :
- *"Faisait-elle autorité à la date X ?"* → `date <= X < (date_overruled or +inf)`

---

## 7. Cas particuliers

| Cas | Traitement |
|---|---|
| Arrêt non publié | `publication = null`, `importance ≥ 2` sauf exception |
| Arrêt de principe ancien (ex : Patureau-Miran 1892) | Marquer `is_principe_historique = true` |
| Arrêt de revirement | Lien `cites_revirement_of` + champ `overrules: [decision_ids]` |
| Arrêt synthétique (M6) | `is_synthetic = true`, ID commence par `SYNTH-` |
| Décision administrative (CE/TA/CAA) | Même schéma, `juridiction = CE/TA/CAA` |

---

## 8. Connexions

- [[Format-Fondement-Juridique]] — pendant pour les codes/articles
- [[Benchmark-KG-Juridique-FR-Design]] — design du benchmark, utilise ce format
- [[Design-Rubrique-Hierarchisee]] — usage des `JPReference` dans les rubriques GT
- `05-Technique/benchmark/schema.py` — implémentation Pydantic à aligner
- [[2026-04-20]] — décision d'introduire `sens_vs_question` et `importance`
