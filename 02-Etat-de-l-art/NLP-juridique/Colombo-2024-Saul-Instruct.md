---
tags: [article, llm, nlp-juridique, francais, open-source]
categorie: "NLP-juridique"
titre_complet: "Saul-Instruct: A Large Language Model dedicated to Law"
auteurs: "Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo, André Martins, Caio Filippo Corro, Fabrizio Esposito, Vera Lucia Raposo"
annee: 2024
type: "Modèle LLM + publication"
venue: "CentraleSupélec / Equall.ai"
url: "https://www.centralesupelec.fr/en/launch-of-the-first-large-language-model-LLM-dedicated-to-law"
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# Saul-Instruct — Colombo et al. 2024

> [!info] Metadonnees
> **Lead** : Pierre Colombo (labo MICS, CentraleSupélec / Paris-Saclay)
> **Startup** : Equall.ai
> **Annee** : mars 2024
> **Paramètres** : 7B
> **Infrastructure** : ADASTRA (supercalculateur français) + GPUs AMD

## Resume

**Premier LLM dédié au droit**, développé par une équipe française (CentraleSupélec) avec focus transjuridictionnel (US + Europe). 7 milliards de paramètres, entraîné sur ADASTRA. Dépasse les autres modèles 7B sur benchmarks juridiques. Disponible pour usage commercial.

## Contributions principales

1. Premier LLM spécialisé juridique venu d'Europe / France
2. Disponibilité commerciale + recherche
3. Validation de l'intérêt des modèles *domain-specific* pour le juridique
4. Infrastructure souveraine (ADASTRA, GPUs AMD)

## Equipe

- **Pierre Colombo** (MICS CentraleSupélec) — lead
- Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver
- Rui Melo (IST Lisbonne)
- André Martins
- **Caio Filippo Corro** (Sorbonne)
- Fabrizio Esposito, Vera Lucia Raposo (NOVA School of Law, Lisbonne)
- Equall.ai (startup Paris)

## Architecture et corpus

- **Taille** : 7B paramètres
- **Base** : probablement Mistral 7B (non confirmé dans source)
- **Corpus** : données transjuridictionnelles (US + Europe) — **focus Europe**
- **Pas de détail** sur la proportion FR / autres langues européennes

## Resultats annonces

> "Outperforms other 7 billion parameter models on legal benchmarks"

*Pas de chiffres précis dans la source consultée — à chercher dans le papier arxiv / HF.*

## Points forts

- **Premier modèle juridique francophone** (relatif — c'est multilingue européen)
- Infrastructure souveraine (ADASTRA)
- Usage commercial autorisé
- Équipe solide (CentraleSupélec + NOVA + Sorbonne)

## Limites

- Pas de focus FR natif — "focus Europe" reste flou
- Pas d'évaluation spécifique sur droit FR national documentée
- 7B seulement — limites de raisonnement sur tâches complexes

## Liens avec mon projet

> [!important] Candidat naturel comme baseline
> Saul-Instruct = **candidat naturel** pour notre Baseline 1/2 (LLM seul / LLM + RAG). Il est français, domain-specific, et 7B → léger à déployer. À inclure dans notre panel de modèles à benchmarker.

### Ce qu'on peut reutiliser
- **Modèle** comme baseline directement (Saul vs Saul + GraphRAG)
- **Approche méthodologique** : fine-tuning domain-specific sur 7B
- **Infrastructure** : ADASTRA est accessible aux chercheurs CentraleSupélec

### Ce qu'il faut creuser
- Vérifier la proportion FR dans le corpus d'entraînement
- Vérifier les benchmarks sur lesquels ils ont été évalués
- Tester sur Les-Audits-Affaires pour comparer à GPT-4o, Gemini

## Connexions

- [[Alhajar-2025-Les-Audits-Affaires]] — bench FR où tester Saul
- [[JuriBERT]] — autre modèle FR juridique (plus petit, encoder)
- [[CamemBERT]] — baseline générique FR
- [[Equall.ai]] — startup à suivre

## Questions ouvertes

- [ ] Saul est-il disponible sur HuggingFace ? Sous quelle licence exacte ?
- [ ] Y a-t-il un papier arxiv associé avec détails d'entraînement ?
- [ ] Corpus FR : quelle proportion du training mix ?
- [ ] Contact possible avec l'équipe Colombo ?

## Ressources

- Article CentraleSupélec : https://www.centralesupelec.fr/en/launch-of-the-first-large-language-model-LLM-dedicated-to-law
- Equall.ai website
- HuggingFace : à confirmer (chercher "Equall/Saul-Instruct-v1")

## Notes personnelles

- **Forte opportunité de collaboration** : Pierre Colombo est à CentraleSupélec, sa démarche et le stage sont proches → contact possible ?
- Saul + GraphRAG pourrait être **notre configuration cible** si Saul s'avère performant
- À comparer systématiquement à Gemma 3, Mistral, Qwen dans notre benchmark
- Le **focus Europe** est à la fois force (multilingue) et faiblesse (pas de spécialisation FR)
