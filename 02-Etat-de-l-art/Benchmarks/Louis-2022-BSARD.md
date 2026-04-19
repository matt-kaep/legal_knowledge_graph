---
tags: [article, benchmark, francais, recherche-information, droit-belge]
categorie: "Benchmarks"
titre_complet: "A Statutory Article Retrieval Dataset in French (BSARD)"
auteurs: "Antoine Louis, Gerasimos Spanakis"
annee: 2022
type: "Article de conférence"
venue: "ACL 2022"
url: "https://aclanthology.org/2022.acl-long.468/"
doi: "10.18653/v1/2022.acl-long.468"
hf_url: "https://huggingface.co/datasets/maastrichtlawtech/bsard"
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# BSARD — Louis & Spanakis 2022

> [!info] Metadonnees
> **Auteurs** : Antoine Louis, Gerasimos Spanakis
> **Institution** : Maastricht Law & Tech Lab (en collaboration avec Droits Quotidiens)
> **Annee** : 2022 (ACL long paper)
> **URL HF** : https://huggingface.co/datasets/maastrichtlawtech/bsard
> **Licence** : CC BY-NC-SA 4.0

## Resume

**Belgian Statutory Article Retrieval Dataset** — premier dataset de recherche d'articles statutaires **en français**. 22 600+ articles belges + 1 100 questions en langage naturel annotées par 6 juristes. Tâche principale : *"pour une question en langage naturel, retrouver les articles de loi pertinents"*. Publication ACL 2022.

## Contributions principales

1. **Premier benchmark de retrieval juridique en français** (belge mais FR)
2. Corpus massif : 22 600+ articles statutaires + 1 100 questions annotées manuellement
3. Questions formulées **en langage naturel** par de vraies personnes (pas construites artificiellement)
4. Dataset public et réutilisable → a inspiré plusieurs travaux dérivés (legal-camembert, GNN retrieval)

## Methodologie

### Donnees

| Élément | Valeur |
|---|---|
| Articles statutaires | 22 600+ |
| Questions totales | ~1 100 |
| Train | 886 questions |
| Test | 222 questions |
| Codes juridiques belges couverts | 32 |
| Langue | Français (belge, `fr-BE`) |
| Taille fichiers | 52.6 MB |

### Schéma des articles

```python
{
  'id', 'article', 'code', 'article_no', 'law_type',
  'book', 'part', 'act', 'chapter', 'section', 'subsection',
  'description'
}
```

> Le schéma est **hiérarchique** (Code → Livre → Partie → Chapitre → Section → Article) — similaire à ce qu'on aura pour Légifrance.

### Schéma des questions

```python
{
  'id', 'question', 'category', 'subcategory',
  'extra_description',
  'article_ids'  # IDs des articles pertinents (ground truth)
}
```

### Exemple concret

```json
{
  "id": "724",
  "question": "La police peut-elle me fouiller pour chercher du cannabis ?",
  "category": "Justice",
  "subcategory": "Petite délinquance",
  "article_ids": "13348"
}
```

### Annotation

- **6 juristes belges** (30-60 ans) pour l'annotation
- Questions issues de Droits Quotidiens (ONG d'accès au droit)
- Collecte : mai 2021

## Resultats de reference

- **Pas de baseline quantifiée** dans la fiche dataset elle-même
- Travail dérivé : [[Louis-Spanakis-2023-Finding-the-Law-GNN]] (arxiv 2301.12847) ajoute un GNN pour améliorer le retrieval statutaire

## Points forts

- **Vrai français** (même si belge) — le vocabulaire juridique partage 90%+ avec le FR
- **Questions authentiques** (Droits Quotidiens) — pas de biais "question écrite par data scientists"
- Annotation par vrais juristes
- **Schéma hiérarchique d'articles** directement réutilisable pour Légifrance
- Plusieurs modèles dérivés disponibles (legal-camembert, legal-distilcamembert)
- Corpus significatif (1 100 questions = 5x plus que Les-Audits-Affaires × 9 codes... euh attends : Les-Audits = 2670 mais c'est des scénarios multi-dimensionnels)

## Limites

- **Droit belge**, pas français → attention à la transposition directe (codes ≠ codes FR)
- **Licence CC BY-NC-SA 4.0** → non commerciale
- Corpus daté de **mai 2021** (potentiellement obsolète)
- Pas d'articles décrets/directives/ordonnances
- **Pas de jurisprudence** — uniquement articles statutaires
- Pas de baseline publié dans la fiche HF

## Liens avec mon projet

> [!important] Hautement pertinent — benchmark de retrieval FR
> BSARD est **le seul benchmark de retrieval juridique en français** (même si belge). Sa structure hiérarchique d'articles et son format question→articles sont **directement transposables à Légifrance**. On peut s'en inspirer très fortement pour construire un équivalent FR national.

### Ce qu'on peut reutiliser
- **Schéma hiérarchique** des articles (Code → Livre → … → Article) — adapter à Légifrance
- **Format question → article_ids** pour notre Module 1 ou 4
- **Méthodologie d'annotation** (6 juristes, questions naturelles)
- **Modèles dérivés** :
  - `maastrichtlawtech/legal-camembert` — CamemBERT fine-tuné juridique FR
  - `maastrichtlawtech/legal-distilcamembert` — version distillée
- **Baselines historiques** à battre sur retrieval (BM25, dense retrieval, GNN)
- **Inspiration pour le GNN retrieval** (arxiv 2301.12847) — directement dans notre ligne GraphRAG

### Ce qu'il faut adapter
- Passer du droit **belge** au droit **français** (Judilibre + Légifrance)
- Licence non commerciale → recréer un équivalent FR avec licence plus permissive
- Ajouter la **dimension jurisprudence** (BSARD n'a que des articles)
- Élargir au-delà des 32 codes (articles + décrets + arrêtés)

### Intégration dans notre benchmark
BSARD peut servir :
- **baseline amont** pour tester des modèles FR juridiques (via les modèles legal-camembert)
- **inspiration méthodologique** pour la structure de notre Module 1 bis (retrieval article FR)
- **pont** vers la communauté académique francophone juridique (Maastricht LawTech → contact possible)

## Connexions

### Articles liés
- [[Louis-Spanakis-2023-Finding-the-Law-GNN]] — extension GNN à BSARD (très proche de notre axe GraphRAG)
- [[Alhajar-2025-Les-Audits-Affaires]] — complément : BSARD = retrieval, Les-Audits = QA
- [[Harvard-LIL-2024-Open-French-Law-RAG]] — benchmark FR RAG (complémentaire)

### Concepts liés
- [[Schéma hiérarchique d'articles juridiques]]
- [[Retrieval statutaire]]
- [[legal-camembert]]

### Questions soulevées
- Peut-on contacter Antoine Louis (Maastricht) pour collaboration ?
- Peut-on reconstruire un BSARD français national (22k articles FR + 1k questions) ?
- Les modèles `legal-camembert` sont-ils transposables au FR français ?

## Citation

```bibtex
@inproceedings{louis2022statutory,
  title = {A Statutory Article Retrieval Dataset in French},
  author = {Louis, Antoine and Spanakis, Gerasimos},
  booktitle = {Proceedings of the 60th ACL},
  year = {2022},
  address = {Dublin, Ireland},
  pages = {6789-6803},
  doi = {10.18653/v1/2022.acl-long.468}
}
```

## Notes personnelles

- **Trouvaille complémentaire majeure** : avec BSARD + Les-Audits-Affaires, on a enfin deux piliers FR réutilisables
- Les modèles **legal-camembert** sont à tester dans notre pipeline de baselines
- **Antoine Louis** est probablement au courant de l'écosystème juridique FR/EU → contact possible via Maastricht
- Leur papier dérivé sur les **GNN + retrieval statutaire** (2023) est **très proche** de notre axe GraphRAG — à lire en priorité
- Licence NC limite la réutilisation commerciale mais pas la recherche académique
