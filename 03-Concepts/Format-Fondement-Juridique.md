---
tags: [concept, format, fondement-juridique, normalisation, benchmark, kg]
type: format-canonique
aliases: [Format-Article, Format-Code, Normalisation-Articles]
domaine: transverse
status: en-vigueur
created: 2026-04-20
modified: 2026-04-20
---

# Format canonique des fondements juridiques (codes + articles)

> Référence unique pour toutes les représentations d'articles et de codes dans le projet : KG, benchmark, ground truth, extraction, évaluation.

Source d'origine : `Hector AI/Tech/judilibre-api/enrichissement_base_complete.ipynb` — pipeline d'enrichissement des 1,125,968 décisions (CC + CA + TJ).

---

## 1. Normalisation des codes

### Slug canonique

Un code est toujours représenté par un **slug lowercase ASCII, kebab_case** (underscore) :

| Nom officiel | Slug canonique |
|---|---|
| Code civil | `code_civil` |
| Code du travail | `code_du_travail` |
| Code de procédure civile | `code_de_procedure_civile` |
| Code de la consommation | `code_de_la_consommation` |
| Code pénal | `code_penal` |
| Code général des impôts | `code_general_des_impots` |
| Code monétaire et financier | `code_monetaire_et_financier` |

### Règles de normalisation

```python
def normalize_code(name: str) -> str:
    s = name.lower().strip()
    s = strip_accents(s)          # "é" → "e", "ô" → "o", etc.
    s = re.sub(r"[''']", "_", s)   # apostrophes → underscore
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s
```

### Codes officiels reconnus

**67 codes officiels** + **8 variantes historiques** (Code Napoléon → Code civil, Nouveau code de procédure civile → Code de procédure civile, etc.). Liste exhaustive dans le notebook `enrichissement_base_complete.ipynb`.

---

## 2. Normalisation des articles

### Format canonique

Un article est représenté par la concaténation `{PREFIX}{NUMBER}` :

| Écriture source | Forme canonique |
|---|---|
| `article 1240` | `1240` |
| `art. L. 122-14-3` | `L122-14-3` |
| `art. R.1234-1` | `R1234-1` |
| `art. D. 4121-1` | `D4121-1` |
| `art. A. 123` | `A123` |
| `1382` (ancien) | `1382` |

### Règles

- **Préfixe** ∈ `{L, R, D, A, E}` — en MAJUSCULE, sans point, sans espace
  - Vide si pas de préfixe (Code civil ancien, Code pénal ancien)
- **Numéro** : chiffres et tirets uniquement, sans espaces ni points
- Ex : `L. 122-14-3` → préfixe `L`, numéro `122-14-3` → forme canonique `L122-14-3`

```python
def normalize_article(prefix: str, number: str) -> str:
    prefix = prefix.upper().strip() if prefix else ""
    num = re.sub(r"[\s.]+", "", number.strip())
    return f"{prefix}{num}" if prefix else num
```

### Alinéas

L'alinéa est **extrait** (regex `(?:\s+al(?:inéa)?\.?\s*(\d+))?`) mais **non intégré** dans la forme canonique par défaut. Si besoin, stocker dans un champ séparé `alinea: int | None`.

---

## 3. Clé canonique `code:article`

La **paire** est la clé de jointure universelle pour le KG et le benchmark :

```
<slug_code>:<article_canonique>
```

Exemples :
- `code_civil:1240`
- `code_du_travail:L122-14-3`
- `code_de_procedure_civile:700`
- `code_penal:222-33`
- `code_general_des_impots:1736`

```python
def make_pair_key(code_slug: str, article_norm: str) -> str:
    return f"{code_slug}:{article_norm}"
```

### Usage

- **KG** : identifiant de nœud `Article`
- **Benchmark** : valeur dans `articles_attendus` de la rubrique (cf. [[Design-Rubrique-Hierarchisee]])
- **Extraction** : clé de dédoublonnage des articles cités dans un arrêt

---

## 4. Extraction automatique depuis texte

Le notebook fournit **3 extracteurs regex** produisant les 3 champs enrichis `codes`, `articles`, `code_article_pairs` :

- `extract_codes(text)` → liste de noms de codes détectés
- `extract_articles(text)` → liste de `{prefix, number}`
- `extract_code_article_pairs(text)` → liste de `{code, article}` co-localisés

### Performance et couverture

Sur la base complète Judilibre enrichie :

| Juridiction | Records | Avec paires | Avec codes |
|---|---|---|---|
| Cour de cassation | 553 075 | 86.2 % | 93.7 % |
| Cours d'appel | 430 654 | 92.7 % | 97.8 % |
| Tribunal judiciaire | 142 239 | 87.2 % | 94.6 % |

> Soit ~88 % de couverture globale des paires `code:article` sur ~1.1 M décisions.

### Zones exploitées dans les arrêts

`introduction`, `visa`, `motivations`, `moyens`, `dispositif`, `expose`, `annexes` — plus le champ `visa` structuré (gold standard pour les articles) et le champ `text` complet.

---

## 5. Intégration avec le schéma Pydantic existant

Dans [[schema.py]] (`05-Technique/benchmark/`), `ArticleReference` doit utiliser le format canonique :

```python
class ArticleReference(BaseModel):
    code: str        # ⟵ DOIT être un slug canonique (code_civil, code_du_travail, ...)
    number: str      # ⟵ DOIT être la forme canonique (1240, L122-14-3, ...)
    alinea: int | None = None
    version_date: date | None = None
    relevance: Literal["central", "supporting"] = "central"
    why: str | None = None

    @property
    def canonical_key(self) -> str:
        return f"{self.code}:{self.number}"
```

> [!warning] Migration
> Le champ `code` du schéma Pydantic actuel accepte du texte libre ("Code du travail"). À normaliser progressivement : ajouter un validator qui appelle `normalize_code()` pour tolérer les entrées en texte brut.

---

## 6. Cas particuliers à surveiller

| Cas | Traitement |
|---|---|
| Article abrogé (ex : 1382 ancien C. civ.) | Garde la clé canonique ; marquer `abrogated_since: date` |
| Recodification (1382 → 1240 en 2016) | Table de correspondance séparée, lien `recodified_as` dans le KG |
| Article inexistant / mal orthographié | Rejet à l'extraction, log pour review manuelle |
| Articles d'annexes (`GIANNEXE IV...`) | Slug distinct : `code_general_des_impots_annexe_iv` |
| Textes non codifiés (Décrets, Ordonnances) | **Hors scope de ce format** — traiter séparément (pas de code) |

---

## 7. Observation empirique — graphe bipartite 2026-04-21

La construction du graphe bipartite JP × Articles (cf. [[Design-Graphes-Phase-AB]]) a révélé l'impact concret du problème de recodification, en particulier sur la réforme du droit des obligations de 2016 :

| Ancien | Nouveau | Citations corpus CC (ancien) |
|---|---|---:|
| `code_civil:1382` | `code_civil:1240` | 1 030 |
| `code_civil:1134` | `code_civil:1103`/`1104`/`1193` | 1 992 |
| `code_civil:1147` | `code_civil:1231-1` | ~700 |

**Conséquence si non traité** :
- Deux arrêts sur la **même règle juridique** (responsabilité délictuelle) mais publiés à des dates différentes ne sont pas connectés dans le graphe via cet article → faux clivage entre "avant 2016" et "après 2016".
- La centralité des articles historiques (1382, 1134) est sous-estimée — ils sont visuellement des hubs importants mais leur "vraie" centralité est fragmentée entre l'ancien et le nouveau numéro.

**Décision d'implémentation recommandée** :
- Option **(b)** de la section §6 — table de correspondance externe + arêtes `recodified_as` ajoutées au graphe lors de la construction.
- **Ne pas** écraser la donnée brute (options (a) et (c) écartées) — fidélité historique requise pour la recherche d'arrêts anciens.

À implémenter dans une future pass d'enrichissement du graphe bipartite (après v1 livrée le 21 avril).

---

## 8. Connexions

- [[Benchmark-KG-Juridique-FR-Design]] — design global du benchmark
- [[Design-Rubrique-Hierarchisee]] — usage dans les rubriques GT
- [[Design-Graphes-Phase-AB]] — graphe bipartite qui a exposé le problème de recodification
- [[Format-Jurisprudence]] — pendant pour la JP
- [[2026-04-20]] — décision d'adopter ce format comme référence unique
- [[2026-04-21]] — validation empirique du problème de recodification
