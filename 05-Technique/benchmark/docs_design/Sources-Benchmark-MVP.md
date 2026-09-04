---
tags: [benchmark, sources, exploration, crfpa, doctrine]
type: inventaire-sources
status: en-cours
created: 2026-04-20
modified: 2026-04-20
---

# Inventaire des sources candidates — benchmark MVP (20-50 questions)

> Exploration du 2026-04-20. Objectif : identifier ~20 sources exploitables pour construire rapidement un MVP 20-50 questions au format [[Design-Rubrique-Hierarchisee]], puis étendre à 300 (cf. planning [[2026-04-20]]).

---

## Découverte majeure : grilles de notation officielles CNB

Depuis la session 2025, suite à un recours administratif, le **Conseil National des Barreaux** publie les **grilles de notation officielles** pour les 12 épreuves écrites du CRFPA. C'est une nouveauté (*"toute première fois que les grilles de notation officielles sont publiées"*) qui change la donne.

**Implication pour le benchmark** : on a la combinaison gold standard sans annotation :
- **Sujet CNB** → définit la question
- **Grille CNB** → définit la rubrique officielle (points attendus, barème → strates `core` / `expected` / `expert`)
- **Meilleures copies IEJ** → confirment articles + JP effectivement utilisés par les candidats top

> Cette combinaison permet de construire des rubriques **validées par le jury national**, ce qui est la meilleure ground truth qu'on puisse espérer sans annotation humaine dédiée.

---

## Sources CRFPA — évaluation détaillée

### 🥇 Tier 1 — Gold (officielles, directement exploitables)

| Source | URL | Contenu | Format | Statut |
|---|---|---|---|---|
| **CNB** — sujets officiels | cnb.avocat.fr/actualite/sujets-des-epreuves-ecrites-de-lexamen-dentree-au-crfpa-2024 | Sujets officiels 12 épreuves (NDS, Oblig, spé, proc) | PDF direct | ✅ Libre |
| **Cap'Barreau** — grilles CNB officielles | capbarreau.com/corriges/ | Grilles de notation officielles 2025 des 12 épreuves | PDF direct | ✅ Libre, provenance CNB |
| **IEJ Strasbourg** — meilleures copies | iej.unistra.fr/examen-dacces-au-crfpa/la-preparation/meilleures-copies-session-2023/ | 12 matières, sessions 2022/2023/2024/2025 | PDF direct | ✅ Libre |

> ⚠️ À vérifier : est-ce que les PDF Strasbourg sont des copies scannées (OCR requis) ou tapées (texte direct) ? À tester sur 1 PDF.

### 🥈 Tier 2 — Utiles (sujets sans corrigés ou corrigés partiels)

| Source | URL | Contenu | Limite |
|---|---|---|---|
| **Mission Avocat** | mission-avocat.fr/examen-crfpa-annales/ | Sujets 2018-2025, toutes matières | ❌ Pas de corrigés |
| **Objectif Barreau** | objectif-barreau.fr/annales-crfpa | Sujets + 2 matières corrigées (Droit obligations 2019-2020) | ⚠️ Corrigés © soumis à usage pédagogique |
| **Prépa Dalloz** | prepa-dalloz.fr/services-gratuits | 1600+ annales, 2 corrigés gratuits (NDS, Grand Oral) | ⚠️ CGV à vérifier |
| **Doc-du-juriste** | doc-du-juriste.com (blog) | Corrigés ponctuels (ex. NDS 2024) | ⚠️ 403 sur fetch direct, à retester |

### 🥉 Tier 3 — Backup

- **lealaw.fr** : corrigés payants (non exploitables MVP)
- **correction-crfpa.fr** : corrections d'années anciennes (2020) — à tester
- **capavocat.fr** : annales en ligne
- **centredeformationjuridique.com** : annales par IEJ

---

## Sources Doctrine — évaluation détaillée

### 🥇 Tier 1 — Libres et structurées

| Source | URL | Forces | Faiblesses |
|---|---|---|---|
| **Actu-Juridique** | actu-juridique.fr | ✅ Libre, équipe pro (juristes + journalistes), JP citée systématiquement, 6 domaines juridiques structurés | ⚠️ CGU à vérifier |
| **Village-Justice** | village-justice.com | ✅ Libre, ~30 484 articles catalogués, volumétrie énorme | ❌ Qualité variable (communauté mixte), citations non systématiques |

### 🥈 Tier 2 — Libres mais hétérogènes

| Source | URL | Notes |
|---|---|---|
| **Légavox** | legavox.fr | Blogs d'avocats, libre, corpus substantiel jusqu'à 2026. Articles de loi rarement cités en home, à vérifier dans le corps |
| **Doc-du-juriste (blog)** | doc-du-juriste.com/blog | Certains articles ouverts, quality variable |
| **OpenEdition Journals** | journals.openedition.org | 672 revues SHS en OA, mais peu de droit FR visible sans recherche ciblée |

### ❌ Sources bloquées à l'exploration

| Source | Raison |
|---|---|
| **Dalloz-Actualité** | 403 sur WebFetch — accès bot bloqué. Section gratuite existe mais nécessite navigateur |
| **HAL-SHS Droit** | Protection anti-bot Anubis → fetch bloqué. À retester via API HAL OAI-PMH |

### 💡 Tier 3 — À explorer manuellement (non testés ici)

- **Lexbase** : section actualité gratuite ?
- **Journal Spécial des Sociétés** : articles libres ?
- **Le Petit Juriste** (lepetitjuriste.fr) : articles étudiants
- **Gazette du Palais** (accès Lextenso limité)
- **Blogs de cabinets** (Hogan Lovells, Gide, CMS Francis Lefebvre...) : publications techniques gratuites
- **Revue générale du droit** (revuegeneraledudroit.eu)

---

## Shortlist 20 sources pour le MVP

### Priorité 1 — démarrer ici (Semaine 17-18)

1. **CNB — sujets officiels CRFPA 2024 + 2025** (12 épreuves × 2 ans = 24 énoncés canoniques)
2. **Cap'Barreau — grilles de notation officielles CNB 2025** (= rubriques gold)
3. **IEJ Strasbourg — meilleures copies 2023 + 2024 + 2025** (matière par matière, sources d'articles+JP réellement cités)

> Ces 3 sources suffiront probablement à produire **les 20-50 premières rubriques** sans scraping complexe. 12 épreuves × 2-3 sessions ≈ 24-36 questions candidates.

### Priorité 2 — extension contrôlée (Semaine 18-19)

4. Mission Avocat (sujets complémentaires 2018-2022)
5. Objectif Barreau (corrigés droit des obligations 2019-2020)
6. Prépa Dalloz (sujets corrigés NDS + Grand Oral)
7. Doc-du-juriste (corrigés ponctuels)
8. IEJ Strasbourg — sessions 2022
9. CNB — sujets 2023, 2022 (si disponibles)
10. Cap'Barreau — éventuelles grilles antérieures

### Priorité 3 — diversification thématique doctrine (Semaine 19-20)

11. Actu-Juridique — section Droit Civil
12. Actu-Juridique — section Droit Administratif
13. Actu-Juridique — section Droit Fiscal
14. Actu-Juridique — section Droit Pénal
15. Actu-Juridique — section Droit des Affaires
16. Village-Justice — articles à fort capital de citations
17. Le Petit Juriste (étudiants, souvent bien sourcé)
18. Blog cabinet Hogan Lovells (technique)
19. Blog cabinet Gide (technique)
20. Revue générale du droit (universitaire libre)

---

## Stratégie d'extraction

### Pour les sources CRFPA (Tier 1)

Pipeline quasi-direct :
```
Sujet CNB (PDF) ──┐
Grille CNB (PDF) ─┼──► LLM ──► Rubrique structurée ──► Relecture rapide
Meilleures copies ┘          (mapping barème → strates)
```

Avantages :
- Format prévisible (PDF structuré)
- Ground truth officielle (pas d'extraction d'opinion)
- 1 question par épreuve × 12 épreuves × 2 ans = 24 questions de base très propres

### Pour la doctrine (Tier 2-3)

Pipeline plus complexe :
```
Article HTML ──► Extraction (titre, question implicite, citations) ──► LLM génère rubrique ──► Relecture + correction
```

À calibrer sur 3-5 articles pour estimer le coût.

---

## Risques et points d'attention

- **Qualité des PDF IEJ Strasbourg** : copies scannées ou texte ? Impact OCR.
- **CGU Cap'Barreau** : on republie des grilles officielles CNB, mais il faut vérifier la licence de republication.
- **Biais CRFPA** : questions tournées "programme d'examen", peut-être peu de cas "praticien" réels. Doctrine compense.
- **Doctrine non structurée** : la qualité des rubriques dépend de la qualité éditoriale variable de chaque article.
- **Contamination LLM** : les annales CRFPA récentes (2023-2025) sont probablement dans les corpus d'entraînement des LLMs → risque de réponses mémorisées. Tester avec une question très récente (2025) et une ancienne pour comparer.

---

## Prochaines étapes concrètes

- [ ] Télécharger 1 sujet CNB + 1 grille Cap'Barreau + 1 meilleure copie IEJ sur même matière
- [ ] Vérifier format PDF (texte ou scan) des meilleures copies IEJ Strasbourg
- [ ] Simuler extraction LLM d'une rubrique sur ce triplet, mesurer temps et qualité
- [ ] Tester fetch Dalloz-Actualité avec un User-Agent réaliste (si possible)
- [ ] Explorer 2-3 articles Actu-Juridique pour voir densité citations
- [ ] Estimer nombre total de questions réellement extractibles du Tier 1

---

## Connexions

- [[Design-Rubrique-Hierarchisee]] — format cible des rubriques
- [[Format-Fondement-Juridique]] / [[Format-Jurisprudence]] — normalisation des citations extraites
- [[2026-04-20]] — journal de décision
- [[Benchmark-KG-Juridique-FR-Design]] — design global
