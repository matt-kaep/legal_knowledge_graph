---
tags: [article, benchmark, droit-francais, droit-affaires]
categorie: "Benchmarks"
titre_complet: "Les-Audits-Affaires: A French Business Law Compliance Benchmark for LLMs"
auteurs: "Mohamad Alhajar, legml.ai"
annee: 2025
type: "Dataset + harness"
venue: "HuggingFace Datasets"
url: "https://huggingface.co/blog/legmlai/les-audits-affaires"
dataset_url: "https://huggingface.co/datasets/legmlai/les-audits-affaires"
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# Les-Audits-Affaires — LegMLAI 2025

> [!info] Metadonnees
> **Auteurs** : Mohamad Alhajar et legml.ai (Paris)
> **Annee** : juin 2025 | **Venue** : HuggingFace Datasets
> **Type** : Benchmark (dataset + evaluation harness)
> **URL blog** : https://huggingface.co/blog/legmlai/les-audits-affaires
> **Dataset** : https://huggingface.co/datasets/legmlai/les-audits-affaires
> **Licence** : à vérifier

## Resume

Premier benchmark spécifiquement dédié à l'évaluation des LLMs en **droit des affaires français**. **2670 cas de test** réels basés sur 400+ personas professionnels, couvrant **9 codes juridiques français**. Méthodologie originale d'évaluation en **5 dimensions** (Action, Délai, Documents, Impact, Conséquences). Source légale ancrée sur Légifrance via le corpus `louisbrulenaudet`.

## Contributions principales

1. **Premier benchmark FR** vraiment adossé au droit français des affaires (Légifrance)
2. **Méthodologie 5 dimensions** opérationnalisable (pas juste accuracy)
3. **Approche persona** (400+ profils professionnels réalistes)
4. **Couverture multi-codes** (9 codes juridiques FR)
5. **Pipeline anti-contamination** (régénération, cross-LLM evaluation)

## Methodologie

### Couverture — les 9 codes juridiques

| Code | Cas | Focus |
|---|---|---|
| Droit Financier | 350 | Banque, AML/CFT, services de paiement |
| Droit Commercial | 320 | Contrats, création d'entreprise, insolvabilité |
| Code Général des Impôts | 310 | TVA, IS, déductions |
| Droit des Assurances | 300 | Polices, sinistres, courtiers |
| Droit Fiscal | 290 | Fiscalité internationale, prix de transfert |
| Droit Consommateur | 290 | RGPD, e-commerce, garanties |
| Droit du Travail | 280 | Contrats, rupture |
| Propriété Intellectuelle | 270 | Brevets, marques, licences |
| Marchés Publics | 260 | Appels d'offres, conformité |
| **Total** | **2 670** | |

### Évaluation 5 dimensions

Chaque cas est noté sur :

1. **Action** : que faut-il faire ?
2. **Délai** : quand le faire ?
3. **Documents** : quels documents produire ?
4. **Impact** : coûts / conséquences financières ?
5. **Conséquences** : pénalités / risques légaux ?

> Très différent d'un QA classique — mesure le côté *opérationnel* du conseil juridique.

### Structure d'un cas

```json
{
  "persona": "Marie, CFO startup...",
  "scenario": "situation juridique spécifique",
  "ground_truth": {
    "action": "...",
    "delai": "...",
    "documents": [...],
    "impact": "...",
    "consequences": "..."
  },
  "legal_refs": ["articles depuis Légifrance"]
}
```

### Anti-contamination

1. Pipeline ouvert — régénération des cas avec personas différents
2. Cross-LLM : GPT-4o génère, autre modèle évalue
3. Mises à jour en temps réel depuis Légifrance
4. Variation : mêmes lois, contextes différents

## Resultats cles (annoncés)

| Métrique | Valeur |
|---|---|
| Taux d'hallucination IA financière (général) | **41%** |
| Hallucinations GPT-4o (raisonnement) | 33-48% |
| Hallucinations Gemini 2.0 Flash | 0.7% |
| Entreprises préoccupées par hallucinations | 77% |

## Points forts

- **Vraiment français** (Légifrance, codes FR)
- **Opérationnel** (5 dimensions ≠ juste answer accuracy)
- **Personas** donnent du réalisme et limite la contamination
- **Open sur HuggingFace** → réutilisable directement
- **Multi-codes** → large couverture du droit des affaires

## Limites

- **Zéro jurisprudence** — uniquement des articles/codes
- **Droit des affaires uniquement** — pas de pénal, civil général, admin
- **Benchmark jeune** (juin 2025) — validation communautaire encore faible
- **Publication blog + HF**, pas de papier académique peer-reviewed
- **Méthodologie d'évaluation** des dimensions non standardisée (comment scorer "impact" ?)

## Liens avec mon projet

> [!important] Hautement pertinent — à utiliser directement
> C'est **le seul benchmark en français basé sur Légifrance** avec une méthodologie rigoureuse. À utiliser comme **baseline de notre benchmark** et comme **source d'inspiration méthodologique** pour la structure des tâches. Nous, on ajoutera la **dimension jurisprudence** qui manque totalement ici.

### Ce qu'on peut reutiliser
- Le **dataset complet** comme premier banc de test (2670 cas prêts à l'emploi)
- La **méthodologie 5 dimensions** pour évaluer nos propres tâches
- Le **pattern persona** (400+ profils) pour générer nos tâches JP
- Le **pipeline anti-contamination** (régénération + cross-LLM)
- Le **harness d'évaluation** open-source (`les-audits-affaires-eval-harness`)

### Ce qu'on complète
- **Ajouter la JP** : pour chaque scénario, retrouver les arrêts pertinents
- **Ajouter les baselines manquantes** : RAG + GraphRAG
- **Ajouter les métriques** : traçabilité, sensibilité, reproductibilité
- **Élargir les domaines** : pénal, civil général, admin

### Integration possible dans notre benchmark

```
Baseline 1 (LLM seul)      → test sur Les-Audits-Affaires tel quel
Baseline 2 (LLM + RAG)     → test sur Les-Audits-Affaires + documents Légifrance
Baseline 3 (LLM multi-step) → idem
Baseline 4 (LLM + RAG + agent) → idem
Cible (LLM + GraphRAG)     → idem + KG
```

## Connexions

### Articles lies
- [[Harvard-LIL-2024-Open-French-Law-RAG]] — complémentaire (QA ouvert, ici ferme)
- [[Guha-2023-LegalBench]] — méthodologie IRAC à adapter pour les dimensions

### Concepts lies
- [[Légifrance]]
- [[Corpus louisbrulenaudet]]
- [[Format-QCM-benchmark-juridique-FR]]

### Questions soulevees
- Peut-on contacter legml.ai pour une collaboration ?
- Peut-on étendre Les-Audits-Affaires avec la dimension JP ?
- Comment scorer automatiquement la dimension "impact" ?

## Ressources

- **Dataset** : [legmlai/les-audits-affaires](https://huggingface.co/datasets/legmlai/les-audits-affaires) (HuggingFace, 2.66k exemples)
- **GitHub harness officiel** : [les-audits-affaires-eval-harness](https://github.com/legml-ai/les-audits-affaires-eval-harness)
- **Website** : legml.ai
- **Corpus Légifrance** : `louisbrulenaudet/legifrance`

### Chargement Python (minimal)

```python
import pandas as pd
# Login via `huggingface-cli login` avant la première requête
df = pd.read_parquet(
    "hf://datasets/legmlai/les-audits-affaires/data/train-00000-of-00001.parquet"
)
```

## Citations cles

> "41% des cas d'usage d'IA financière présentent des hallucinations."

> "The AI generates false regulatory deadlines, invents pseudo-declarations, or applies incorrect penalty rates."

## Notes personnelles

- **Trouvaille majeure** : c'est exactement le type de benchmark qu'il nous faut
- L'approche en 5 dimensions est très opérationnelle — correspond à ce qu'un juriste ferait vraiment
- **Contact legml.ai** : potentiel partenariat (ils sont à Paris, le contexte est idéal)
- Leur benchmark = **baseline gratuite** pour tester nos modèles avant de construire notre propre benchmark JP
- **Gap que nous comblons** : eux = articles uniquement ; nous = articles + JP + graphe
