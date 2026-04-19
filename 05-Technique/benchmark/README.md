# Benchmark KG juridique FR — code

Code source du benchmark décrit dans
`05-Technique/methodologies/Benchmark-KG-Juridique-FR-Design.md`.

## Structure

```
benchmark/
├── README.md
├── requirements.txt
├── schema.py              # Pydantic : CaseContext, Decision, M6Annotation, TestCase
├── loaders/
│   ├── les_audits.py      # chargement Les-Audits-Affaires (LegMLAI)
│   └── bsard.py           # chargement BSARD (Maastricht)
└── m6_mvp/
    ├── gold_cases.py      # 5 cas gold standard pour Module 6 (interprétation d'arrêt)
    └── display_mvp.py     # runner qui imprime les cas
```

## Installation

```bash
cd 05-Technique/benchmark
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Accès HuggingFace (pour Les-Audits-Affaires)
huggingface-cli login
```

## Utilisation

### Afficher les 5 cas gold du MVP M6
```bash
python -m m6_mvp.display_mvp
```

### Explorer Les-Audits-Affaires
```bash
python -m loaders.les_audits
```

### Explorer BSARD
```bash
python -m loaders.bsard
```

## Modules couverts dans ce MVP

- ✅ **M6 — Interprétation d'arrêt** : 5 cas gold standard (synthétiques, 5 spécialisations)
- 🔨 M1 — Principe QA : loader Les-Audits prêt
- 🔨 Retrieval : loader BSARD prêt
- ⬜ M2, M3, M4, M5 : à venir

## Notes

- Les **décisions de M6** sont **synthétiques** — faits inventés, inspirés de vrais
  patterns jurisprudentiels mais ne correspondent à aucune affaire réelle.
  Objectif : valider la méthodologie sans risque de biais / copyright.
- Les vraies décisions viendront de Judilibre au prochain sprint.
