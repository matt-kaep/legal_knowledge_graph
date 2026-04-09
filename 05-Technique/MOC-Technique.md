---
tags: [moc, technique]
---

# Stack technique & Prototypes

## Decisions architecturales
```dataview
TABLE statut, date_decision
FROM "01-Projet"
WHERE contains(tags, "decision")
SORT date_decision DESC
```

## Stack envisagee

| Composant | Outil | Statut | Notes |
|---|---|---|---|
| Graph DB | Neo4j / ? | A decider | |
| Ontologie | ELI + custom | A decider | |
| Extraction NER | JuriBERT / ? | A tester | |
| KG Construction | LLM pipeline | A concevoir | |
| Agent | Claude / ? | A concevoir | |
| Embeddings | Legal CamemBERT | A tester | |
| RAG | GraphRAG / ? | A decider | |

## Prototypes
```dataview
TABLE status, date
FROM "05-Technique/prototypes"
SORT date DESC
```
