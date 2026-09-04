---
tags: [benchmark, methodologie, design, rapprochements, judilibre, jp, graph]
type: design-document
status: mvp-α
created: 2026-04-20
modified: 2026-04-20
---

# Design du benchmark rapprochements JP (MVP α)

> Second benchmark complémentaire à [[Design-Rubrique-Hierarchisee]] (benchmark CRFPA).
> Exploite exclusivement la donnée déjà enrichie par `enrichissement_base_complete.ipynb`
> ([[Format-Fondement-Juridique]]).

---

## 1. Problème à résoudre

Le benchmark CRFPA ([[Recap-MVP-2026-04-20]]) mesure la qualité d'une réponse à une question juridique en termes de raisonnement structuré (articles + JP + points de raisonnement). Il ne mesure **pas** la capacité à **naviguer entre arrêts liés** — exactement ce qu'un KG juridique doit permettre.

Or la Cour de cassation publie elle-même, pour chaque arrêt, une liste de "rapprochements" : d'autres arrêts à mettre en regard (confirmation, revirement, précision, extension thématique). Cette liste est dans le champ `rapprochements` des records Judilibre.

**Ces rapprochements constituent une ground truth institutionnelle gratuite** pour évaluer la capacité d'un système à comprendre la filiation entre arrêts — au-delà de la simple similarité sémantique.

---

## 2. Tâche évaluée

**Input** (fourni au LLM) :
- Référence identifiante de l'arrêt : chambre, date, n° de pourvoi, ECLI
- Texte intégral de l'arrêt (champ `text`)
- (Métadonnées : codes, articles extraits du texte)

**Output attendu** (JSON strict) :

```json
{
  "rapprochements": [
    {
      "reference": "Civ. 1re, 14 janvier 2016",
      "pourvoi": "15-13.263",
      "chamber": "Civ. 1re",
      "date": "2016-01-14"
    }
  ]
}
```

**Consigne-type du prompt** :
> "Voici un arrêt de la Cour de cassation. Donne-moi les arrêts qu'il faut **mettre en regard** (rapprochements) pour comprendre sa place dans la ligne jurisprudentielle : arrêts qui confirment, qui contredisent, qui approfondissent ou qui couvrent le même thème."

---

## 3. Ground truth

Source : champ `rapprochements` de chaque record Judilibre.

**Problème** : les champs structurés (`id`, `number`, `url`, `date`) sont **100 % `null`** ; seul le `title` est rempli, sous la forme :

```
"Crim., 18 janvier 2011, pourvoi n° 10-87.525, Bull. crim. 2011, n° 7 (cassation sans renvoi)."
```

**Solution** : regex sur le title → extraction du numéro de pourvoi → normalisation → matching ensembliste.

- **80 % des titles** sont parsables (pourvoi extractible) sur les arrêts récents.
- **~20 % non parsables** : vieux arrêts (bulletin antique) ou format légèrement variant. Exclus de la GT pour cette v1 — à documenter comme biais connu (un LLM qui citerait correctement par mémoire doctrinale serait pénalisé sur ces 20 %).
- **Bonus** : un sous-ensemble des rapprochements récents embarque déjà l'ID Judilibre dans une balise HTML `<a href="...">` — **pré-résolu gratuitement**.

Résolution d'ID : index inverse `pourvoi_normalisé → id_judilibre` construit sur l'ensemble du corpus CC (pass 2). Permet d'enrichir chaque GT avec ECLI, chambre réelle, date — pratique pour analyses fines plus tard.

---

## 4. Sampling

- **Juridiction** : Cour de cassation uniquement
  (CA et TJ n'ont quasi pas de rapprochements — vérifié par sondage).
- **Critère de sélection** : arrêt avec **≥3 rapprochements parsables**
  (≥3 pour que le recall ait du signal ; moins → GT trop petite).
- **Pas de filtre publication** : on garde publiés et non-publiés
  (décision MVP 2026-04-20 — prendre le max de matière).
- **Stratification chambres** : on exporte **tout** (pas de cap artificiel), on rapporte la distribution par chambre dans les stats.

---

## 5. Scoring

### Matching

Normalisation du numéro de pourvoi :
- Strip points, espaces → `10-87.525` / `10.87.525` / `10-87-525` → clé canonique `10-87525`.
- Comparaison ensembliste entre `GT` et `LLM_output`.

### Métriques

- **Precision** = |GT ∩ LLM| / |LLM| → **taux d'hallucination** si < 1.
- **Recall** = |GT ∩ LLM| / |GT| → **couverture des rapprochements officiels**.
- **F1** = harmonique des deux.

### Breakdown optionnel (v1.1)

- Recall par chambre (Crim. vs Civ. 1re vs …) → variance inter-matière.
- Recall par âge d'arrêt (≤5 ans, 5-15 ans, >15 ans) → proxy de la contamination d'entraînement.

---

## 6. Variantes reportées à plus tard (hors MVP)

| Variante | Description | Intérêt |
|---|---|---|
| **β — typage LLM** | Classifier chaque paire (X, rapp) en {confirme, contredit/revirement, approfondit, même thème} via un LLM juge | Évaluer la compréhension fine de la filiation |
| **γ — typage signal faible** | Utiliser la mention `(rejet) / (cassation)` dans le title + chronologie + mots-clés pour typer sans LLM | Proxy gratuit de β |
| **Contrôle contamination** | Variante input = référence seule (sans texte) | Mesurer la mémoire vs la compréhension |
| **Inverse** | Input = article → liste des JP-clés interprétatives | Complémentaire, nécessite une autre GT |

---

## 7. Pipeline d'implémentation

Script unique : `build_rapprochement_benchmark.py` (approche 1 retenue — monolithique).

```
Pass 1 (scan streaming)
  ├─ parse chaque record JSONL
  ├─ regex sur rapprochements[i].title → pourvoi_norm
  └─ garder si |parsables| ≥ 3

Pass 2 (index inverse)
  ├─ re-scan tout le corpus
  └─ {pourvoi_norm: {id, ecli, chamber, date, solution}}

Pass 3 (résolution + export)
  ├─ pour chaque candidat, résoudre ses rapp via l'index
  ├─ canonicaliser la chambre
  └─ dump JSON
```

Coût RAM : ~200 Mo pour l'index (> stocker tout le corpus en mémoire : 5 Go).
Coût CPU : 2 pass × 5 Go streaming ≈ 3-5 min sur Mac M1.

---

## 8. Format de sortie

Fichier : `data/rapprochements/benchmark-rapp-v1.json`

```json
{
  "version": "v1-2026-04-20",
  "variant": "α — non typé",
  "source": "database-judilibre-enrichie/Cour de cassation",
  "filters": {"min_parsable_rapp": 3, "publication_filter": null},
  "stats": {"questions_total": ..., "by_chamber": {...}, "resolution": {...}},
  "questions": [
    {
      "id": "rapp-Q00001",
      "decision": {
        "id": "...", "ecli": "...", "pourvoi": "...",
        "chamber": "Civ. 1re", "decision_date": "...",
        "solution": "...", "publication": [...],
        "codes": [...], "code_article_pairs": [...],
        "text": "..."
      },
      "ground_truth": {
        "rapprochements": [
          {
            "pourvoi": "15-13263",           // normalisé
            "pourvoi_raw": "15-13.263",
            "chamber_hint": "Civ. 1re",
            "raw_title": "Civ. 1re, 14 janvier 2016, pourvoi n° 15-13.263, ...",
            "resolved": {"id": "...", "ecli": "...", "chamber": "...", "date": "..."}
          }
        ],
        "n_parsable": 5,
        "n_total_brut": 7
      }
    }
  ]
}
```

---

## 9. Limites et biais connus

- **Contamination d'entraînement** : les arrêts publiés + récents sont probablement dans les corpus LLM. Mesurable via variante contrôle (cf. §6).
- **Exclusion des 20 % non parsables** : peut sous-estimer la GT pour des arrêts anciens. À documenter dans les stats.
- **Rapprochements non typés** : α ne distingue pas "revirement" vs "même thème". Discriminant pertinent mais agnostique au type de lien.
- **Asymétrie juridictionnelle** : uniquement CC ; pas d'évaluation sur CA / TJ qui n'ont pas la donnée.
- **Effet "arrêt isolé"** : si un arrêt a des rapprochements que la Cass n'a pas (encore) publiés, le LLM peut sortir des JP "correctes" mais non listées → faux négatifs sur la precision. C'est le coût de la parcimonie de la GT institutionnelle.

---

## 10. Prochaines étapes

1. Exécuter le script, vérifier les stats par chambre.
2. Inspecter manuellement 5 questions pour valider la GT.
3. Lancer une première baseline sans RAG (B1 du plan d'eval) sur un sous-échantillon.
4. Implémenter la variante β (typage automatisé) si le signal α est informatif.

---

## Connexions

- [[Design-Rubrique-Hierarchisee]] — benchmark CRFPA (complémentaire)
- [[Format-Fondement-Juridique]] — pair_key articles
- [[Format-Jurisprudence]] — format JP canonique
- [[Recap-MVP-2026-04-20]] — état des lieux MVP CRFPA
- [[Benchmark-KG-Juridique-FR-Design]] — vue d'ensemble
- Script : `build_rapprochement_benchmark.py`
- Output : `data/rapprochements/benchmark-rapp-v1.json`
