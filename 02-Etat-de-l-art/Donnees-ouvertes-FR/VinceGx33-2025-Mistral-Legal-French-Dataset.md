---
tags: [dataset, francais, fine-tuning, reasoning, chain-of-thought]
categorie: "Donnees-ouvertes-FR"
titre_complet: "Mistral Legal French Dataset"
auteurs: "VinceGx33"
annee: 2025
type: "Dataset (HuggingFace)"
venue: "HuggingFace Hub"
url: "https://huggingface.co/datasets/VinceGx33/mistral-legal-french-dataset"
licence: "Apache 2.0"
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# Mistral Legal French Dataset — VinceGx33 2025

> [!info] Metadonnees
> **Publisher** : VinceGx33
> **Date** : octobre 2025
> **Taille** : 14 875 exemples
> **Format** : JSONL (ChatML)
> **Licence** : Apache 2.0
> **URL** : https://huggingface.co/datasets/VinceGx33/mistral-legal-french-dataset

## Resume

Dataset d'entraînement pour fine-tuning de Mistral-7B sur le **droit français**. **14 875 exemples** 100% en français, mixant QA simples (LegalKit, 10k) et raisonnement structuré (COT, 4875). Les COT sont générés depuis des **décisions du Tribunal Judiciaire** via la pipeline Judilibre/Jurica, en 4 parties syllogistiques : situation / cadre applicable / analyse / réponse.

## Structure

| Composant | Exemples | % | Type | Format |
|---|---|---|---|---|
| **LegalKit** | 10 000 | 67.2% | QA factuelles sur le droit FR | user/assistant |
| **COT (Chain-of-Thought)** | 4 875 | 32.8% | Raisonnement juridique structuré | system/user/assistant |
| **Total** | 14 875 | 100% | | ChatML |

### Structure COT — 4 parties (syllogisme FR)

1. **Situation juridique**
2. **Cadre juridique applicable**
3. **Analyse et conditions**
4. **Réponse et conseils**

> C'est littéralement le **syllogisme juridique français** opérationnalisé sous forme de dataset.

## Couverture juridique

- **63 284 références d'articles** cités (6 970 articles uniques)
- **56 termes juridiques spécialisés** détectés
- **75.1%** des exemples contiennent du vocabulaire juridique
- **Top articles cités** : Article 700 (frais de procédure), Article 696 (procédure civile), Article 514…

## Sources

### LegalKit (10 000)
- Source : `louisbrulenaudet/legalkit`
- QA factuelles sur le droit français
- Questions-réponses directes

### COT (4 875)
- Source : `judilibre/jurica-tribunal_judiciaire`
- **Filtrage** : 32 000 meilleures décisions sélectionnées sur 800 000+
- **Critères qualité** : ≥1 500 chars de faits, ≥8 000 chars de raisonnement
- **Génération** : Qwen2.5-7B-Instruct-4bit (local MLX, ~102h M4 Pro)
- **Taux de succès** : 73%

## Methodologie remarquable

### Curriculum Learning
- Exemples 1-10 000 : LegalKit (facile, factuel)
- Exemples 10 001-14 875 : COT (difficile, raisonnement)
- **Pas de shuffle** — ordre maintenu intentionnellement
- Justification : +15-20% perf vs ordre aléatoire (d'après papiers cités)

### Validation structurelle
- ✅ 100% valid JSON
- ✅ 99.98% structure COT correcte
- ✅ 0 message vide

## Usage annoncé

### ✅ Convient pour
- Fine-tuning Mistral-7B-Instruct-v0.3 (cas d'usage principal)
- Adaptation domaine juridique FR
- Instruction following (ChatML)
- Reasoning tasks (via COT)
- Connaissance factuelle (via LegalKit)

### ❌ Ne convient PAS pour
- **Évaluation / benchmarking** (trop domain-specific, pas de ground truth stable)
- RAG (mieux utilisé comme référence que données d'entraînement)
- Pré-entraînement (trop petit, spécialisé)

## Configuration recommandée

```yaml
Base: mistralai/Mistral-7B-Instruct-v0.3
GPU: T4 Medium ($0.60/h)
LoRA: rank=16, alpha=32, dropout=0.05
Training: 3 epochs, batch_size=4, lr=2e-4
Durée: 2-3h, Coût: ~$1.20-1.80
```

## Liens avec mon projet

> [!important] Dataset clé pour notre volet fine-tuning (Axe 1)
> Ce dataset est **exactement** ce qu'il nous faut pour implémenter la composante "fine-tuning léger" de notre analyse comparative des LLMs open-source. Structure COT en 4 parties = syllogisme FR = cohérent avec notre slide 2.

### Ce qu'on peut reutiliser
- **Dataset tel quel** pour fine-tuner Mistral / Gemma / Llama / Qwen sur le droit FR
- **Structure COT 4 parties** comme modèle pour générer nos propres tâches de raisonnement
- **Méthodologie de filtrage** (1500/8000 chars min) pour sélectionner des arrêts de qualité
- **Pipeline de génération COT** via Qwen local pour produire plus de données si besoin

### Limites à anticiper
- **Pas un benchmark** — pas d'évaluation, pas de ground truth externe
- Corpus : **Tribunal judiciaire** (pas Cassation) → pas la même strate hiérarchique
- Format ChatML → à adapter si on utilise un autre format de prompt
- Aucune trace de KG / graphe — purement texte

### Integration dans notre plan benchmark

```
Baseline 1 : Mistral-7B vanilla                    → benchmark
Baseline 1-FT : Mistral-7B + fine-tune sur ce dataset → benchmark
                                                    → mesurer le delta
```

## Connexions

### Articles / ressources liés
- [[Colombo-2024-Saul-Instruct]] — autre piste de modèle FR juridique
- [[Alhajar-2025-Les-Audits-Affaires]] — benchmark pour évaluer un modèle fine-tuné avec ce dataset
- [[louisbrulenaudet/legalkit]] — source amont
- [[judilibre]] — source amont (Jurica)

### Concepts liés
- [[Fine-tuning LoRA]]
- [[Chain-of-Thought (COT)]]
- [[Curriculum Learning]]
- [[Syllogisme juridique francais]]

### Questions soulevées
- [ ] La qualité de la génération COT par Qwen 7B est-elle suffisante ? (73% succès seulement)
- [ ] Peut-on enrichir avec des arrêts de Cassation (au lieu de TJ uniquement) ?
- [ ] Comment évaluer l'amélioration apportée par ce fine-tuning ?

## Citation

```bibtex
@misc{mistral_legal_french,
  author = {VinceGx33},
  title = {Mistral Legal French Dataset},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/datasets/VinceGx33/mistral-legal-french-dataset}
}
```

## Notes personnelles

- **La structure COT en 4 parties = or pur** pour notre projet — c'est exactement la décomposition qu'on veut mesurer
- **À tester en priorité** : fine-tuner Mistral-7B avec ce dataset, puis évaluer sur Les-Audits-Affaires → mesurer le gain
- Le **pipeline de génération COT via Qwen local** est reproductible chez nous (M4 Pro suffit)
- **Contact possible avec VinceGx33** pour comprendre le pipeline de filtrage plus précisément
- Piste : **étendre ce dataset avec des arrêts de Cassation** et notre propre structure KG en annotation
