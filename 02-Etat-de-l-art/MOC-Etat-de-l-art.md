---
tags: [moc, etat-de-l-art]
created: 2026-04-09
modified: 2026-04-09
---

# Etat de l'art — Map of Content

## Statistiques

```dataview
TABLE length(rows) as "Nb articles"
FROM "02-Etat-de-l-art"
WHERE tags AND contains(tags, "article")
GROUP BY categorie
```

### Progression de lecture

```dataview
TABLE
  length(filter(rows, (r) => r.status = "lu")) as "Lus",
  length(filter(rows, (r) => r.status = "en-cours")) as "En cours",
  length(filter(rows, (r) => r.status = "a-lire")) as "A lire"
FROM "02-Etat-de-l-art"
WHERE tags AND contains(tags, "article")
GROUP BY categorie
```

---

## Par thematique

### KG dans le domaine juridique (9 articles)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/KG-juridique"
SORT pertinence ASC, annee DESC
```

### Ontologies & Standards (7 refs)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Ontologies-standards"
SORT pertinence ASC, annee DESC
```

### GraphRAG (8 articles)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/GraphRAG"
SORT pertinence ASC, annee DESC
```

### Construction de KG avec LLMs (3 articles)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Construction-KG-LLM"
SORT pertinence ASC, annee DESC
```

### NLP Juridique & NER (7 articles)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/NLP-juridique"
SORT pertinence ASC, annee DESC
```

### Agents LLM Juridiques (4 articles)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Agents-LLM"
SORT pertinence ASC, annee DESC
```

### Prediction & Citations (5 articles)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Prediction-citations"
SORT pertinence ASC, annee DESC
```

### Donnees ouvertes FR (5 refs)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Donnees-ouvertes-FR"
SORT pertinence ASC, annee DESC
```

### Benchmarks & Evaluation (2 refs)

```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Benchmarks"
SORT pertinence ASC, annee DESC
```

---

## Vues transversales

### Articles a lire en priorite (pertinence haute)

```dataview
TABLE auteurs as Auteurs, categorie as Cat, annee as Annee
FROM "02-Etat-de-l-art"
WHERE status = "a-lire" AND pertinence = "haute"
SORT annee DESC
```

### En cours de lecture

```dataview
TABLE auteurs as Auteurs, categorie as Cat
FROM "02-Etat-de-l-art"
WHERE status = "en-cours"
SORT file.mtime DESC
```

### Tous les articles lus

```dataview
LIST
FROM "02-Etat-de-l-art"
WHERE status = "lu"
SORT annee DESC
```
