---
tags: [methodologie, knowledge-management, llm, a-evaluer]
type: "technique"
statut: "documentee"
origine: "Andrej Karpathy et communaute (2025-2026)"
created: 2026-04-10
modified: 2026-04-10
---

# LLM Wiki — Pattern de construction de base de connaissances par LLM

> Technique de stockage et de maintenance de connaissances ou le LLM construit et maintient incrementalement un wiki persistant a partir de sources brutes. Documentee ici pour evaluation, pas encore implementee.

## 1. L'idee centrale

### Le probleme qu'elle resout

Les systemes RAG classiques (ChatGPT file upload, NotebookLM) fonctionnent de la maniere suivante : on uploade un corpus, le LLM retrouve des chunks pertinents a chaque requete, et genere une reponse. Le probleme : **le LLM redecouvre la connaissance a chaque question**. Rien ne s'accumule. Pour une question subtile qui demande de synthetiser cinq documents, le LLM doit retrouver et assembler les fragments a chaque fois. Rien n'est construit au fil du temps.

### La proposition

Au lieu de simplement retrouver des chunks a la volee, le LLM construit et maintient incrementalement un **wiki persistant** — une collection structuree et interliee de fichiers markdown qui s'interpose entre l'utilisateur et les sources brutes.

Quand une nouvelle source est ajoutee :
1. Le LLM la lit
2. Il en extrait les informations cles
3. Il **integre** ces informations dans le wiki existant
4. Il met a jour les pages d'entites, revise les resumes, note les contradictions, renforce la synthese

**La connaissance est compilee une fois puis maintenue a jour, pas re-derivee a chaque requete.**

### La difference cle

> "The wiki is a persistent, compounding artifact. The cross-references are already there. The contradictions have already been flagged. The synthesis already reflects everything you've read."

Le wiki devient un artefact **compoundant** : chaque nouvelle source et chaque nouvelle question l'enrichit.

### Le partage de travail humain / LLM

- **L'humain** : curation des sources, exploration, bonnes questions, direction strategique
- **Le LLM** : resumes, cross-references, filing, bookkeeping — "everything else"

L'analogie proposee : **Obsidian est l'IDE, le LLM est le programmeur, le wiki est la codebase.**

## 2. Architecture en 3 couches

```
┌─────────────────────────────────────────┐
│  SCHEMA  (CLAUDE.md)                    │
│  Regles du jeu : conventions, workflows │
│  Co-evolue avec l'utilisateur           │
└─────────────────────────────────────────┘
              ^
              │ guide
              v
┌─────────────────────────────────────────┐
│  WIKI  (markdown files)                 │
│  Ecrit par le LLM uniquement            │
│  Summaries, entity pages, syntheses     │
│  Cross-references, index                │
└─────────────────────────────────────────┘
              ^
              │ lit
              v
┌─────────────────────────────────────────┐
│  RAW SOURCES                            │
│  Immuables : PDFs, articles, datasets   │
│  "Source of truth"                      │
└─────────────────────────────────────────┘
```

### Couche 1 : Raw sources
- Articles, papers, datasets, images, transcripts
- **Immuables** : le LLM lit mais ne modifie jamais
- Source of truth

### Couche 2 : Le wiki
- Repertoire de fichiers markdown generes par le LLM
- Types de pages : summaries, entity pages, concept pages, comparisons, syntheses
- Le LLM **possede entierement cette couche** : creation, mise a jour, cross-references
- L'humain lit, ne modifie pas

### Couche 3 : Le schema (CLAUDE.md / AGENTS.md)
- Document qui explique au LLM :
  - Comment le wiki est structure
  - Les conventions
  - Les workflows a suivre (ingest, query, lint)
- **Co-evolue** avec l'utilisateur au fur et a mesure qu'on decouvre ce qui marche
- "C'est ce qui fait du LLM un mainteneur de wiki discipline plutot qu'un chatbot generique"

## 3. Les trois operations

### Ingest — ajouter une source

Flux typique :
1. L'utilisateur depose une source dans `raw/`
2. Le LLM la lit
3. Discussion avec l'utilisateur sur les takeaways
4. Le LLM ecrit une page de resume dans le wiki
5. Met a jour l'index
6. Met a jour **10-15 pages** d'entites et de concepts concernes
7. Ajoute une entree au log

**Variante** : ingest un-par-un (supervise) ou batch (moins supervise).

### Query — poser une question

1. L'utilisateur pose une question
2. Le LLM cherche dans l'index
3. Lit les pages pertinentes
4. Synthese avec citations
5. Sortie sous differentes formes : markdown, tableau, slide deck (Marp), chart (matplotlib), canvas
6. **Point cle** : les bonnes reponses peuvent etre **refilees** dans le wiki comme nouvelles pages

> "This way your explorations compound in the knowledge base just like ingested sources do."

### Lint — verification sante

Periodiquement, demander au LLM de verifier :
- Contradictions entre pages
- Claims obsoletes (supersedees par nouvelles sources)
- Pages orphelines (pas de liens entrants)
- Concepts importants mentionnes mais sans page dediee
- Cross-references manquantes
- Trous de donnees a combler via web search
- Nouvelles questions a investiguer

## 4. Fichiers speciaux

### `index.md` — content-oriented
- Catalogue de **toutes** les pages du wiki
- Chaque page : lien + resume d'une ligne + metadata optionnelle
- Organise par categorie (entites, concepts, sources...)
- Mis a jour a chaque ingest
- Le LLM le lit en premier quand il repond a une question (remplace RAG embeddings a petite echelle)

> "This works surprisingly well at moderate scale (~100 sources, ~hundreds of pages) and avoids the need for embedding-based RAG infrastructure."

### `log.md` — chronological
- Append-only
- Entrees : ingests, queries, lint passes
- **Astuce** : chaque entree commence par un prefixe consistent, ex :
  ```
  ## [2026-04-10] ingest | Zhang et al. 2025 GraphRAG Survey
  ## [2026-04-10] query | Quels benchmarks juridiques FR existent ?
  ## [2026-04-09] lint | Orphans detected: 3 pages
  ```
- Parsable avec des outils unix simples :
  ```bash
  grep "^## \[" log.md | tail -5  # 5 dernieres entrees
  ```

## 5. Outils complementaires

### Obligatoires / quasi-obligatoires
- **Obsidian** comme IDE frontend (graph view, backlinks, Dataview)
- **Git** pour versioning automatique

### Optionnels mais utiles
- **Obsidian Web Clipper** (extension navigateur) pour convertir articles web en markdown
- **Attachment folder fixe** (`raw/assets/`) + hotkey "Download attachments for current file" pour telecharger les images localement
- **Marp** pour generer des slides depuis du markdown
- **Dataview** pour requetes dynamiques sur frontmatter
- **qmd** (local search engine markdown, BM25/vector hybrid, CLI + MCP server) pour scale au-dela de l'index file

## 6. Pourquoi ca marche

> "The tedious part of maintaining a knowledge base is not the reading or the thinking — it's the bookkeeping."

- Updating cross-references
- Keeping summaries current
- Noting contradictions
- Maintaining consistency

Les humains abandonnent les wikis parce que le **cout de maintenance croit plus vite que la valeur**. Les LLMs ne se fatiguent pas, n'oublient pas de mettre a jour un cross-reference, peuvent toucher 15 fichiers en une passe. Le wiki reste maintenu parce que le **cout de maintenance est proche de zero**.

## 7. Filiation conceptuelle

- **Memex de Vannevar Bush (1945)** — vision d'un knowledge store personnel, curate, avec des trails associatifs entre documents. Bush imaginait le web tel qu'il aurait du etre : prive, activement curate, ou les connexions sont aussi importantes que les documents. La partie qu'il ne savait pas resoudre : **qui fait la maintenance**. Le LLM la resout.

- **Zettelkasten** (Niklas Luhmann) — fiches interliees, pensees atomiques, cross-references denses. Meme logique, mais entierement manuelle.

- **Fan wikis** (ex: Tolkien Gateway) — des milliers de pages interliees maintenues par une communaute de volontaires pendant des annees. Le LLM Wiki permet de faire la meme chose **individuellement**.

---

## 8. Mapping avec notre projet actuel

### Ce qu'on a deja qui correspond au pattern

| Element du pattern | Dans notre vault | Note |
|---|---|---|
| Wiki : summaries | `02-Etat-de-l-art/*/` fiches d'articles | Deja en place |
| Wiki : concept pages | `03-Concepts/` | Vide pour l'instant |
| Wiki : syntheses | `06-Analyses/syntheses/` | Commence |
| Schema | `CLAUDE.md` | Existe, minimal |
| MOC / index partiel | `00-HOME.md`, `02-Etat-de-l-art/MOC-Etat-de-l-art.md` | Via Dataview |
| Chronological log | `01-Projet/journal/` | Par jour, pas par operation |

### Ce qui manquerait pour etre "compliant"

| Element | Manque |
|---|---|
| Raw sources dedies | Pas de `raw/` dedie (PDFs restent externes) |
| `index.md` content-oriented | Les MOCs sont thematiques, pas un index global |
| `log.md` chronologique parsable | Le journal par jour n'a pas le format "operation" |
| Workflows explicites dans CLAUDE.md | Les workflows ingest/query/lint ne sont pas documentes |
| Discipline "LLM writes, user reads" | Pas enforcee dans notre workflow actuel |

---

## 9. Evaluation de pertinence pour le projet

### Forces du pattern pour notre cas

1. **Notre corpus grandit par vagues** : 50 papiers a lire, puis ajouts au fil du temps. Le pattern est fait pour ca.
2. **Nous avons deja la plupart des briques** : Obsidian + Dataview + CLAUDE.md + fiches + synthese. L'infrastructure existe.
3. **Le bookkeeping est deja un probleme** : maintenir les cross-references entre 50+ fiches est exactement le type de maintenance fastidieuse que le LLM peut automatiser.
4. **Obsidian + git versioning** : deja utilise, gratuit.
5. **L'approche "compounding artifact"** match notre besoin de construire progressivement une comprehension de l'etat de l'art puis la transformer en decisions de design.

### Faiblesses / risques pour notre cas

1. **Coherence avec notre structure existante** : on a deja une arborescence thematique (00-09) qui ne correspond pas exactement au modele raw/wiki/schema. Forcer un refactor pourrait casser ce qui marche.
2. **Discipline "LLM only writes the wiki"** : dans un projet de recherche, l'humain veut souvent annoter ses propres pensees dans les fiches. Distinguer wiki LLM vs notes humaines demande de la rigueur.
3. **Scale** : on est a ~60 pages. Le pattern est dimensionne jusqu'a ~100 sources / quelques centaines de pages. On peut scale, mais au-dela il faudrait qmd ou similaire.
4. **Query → refile dans le wiki** : demande un reflexe nouveau. Facile d'oublier.
5. **Lint regulier** : demande de la discipline recurrente que les humains oublient — mais c'est aussi ce que le pattern promet de resoudre si on l'automatise.
6. **Compatibilite avec le memoire final** : le pattern est pour explorer et accumuler. Le memoire final (`07-Redaction/`) a ses propres contraintes (format impose, style academique). Il faut garder les deux distincts.

### Recommandation

**A adopter partiellement**, avec prudence :

- **Facile a adopter** (faible cout, haute valeur) :
  - Documenter explicitement les workflows ingest/query/lint dans CLAUDE.md
  - Adopter le principe "bonnes reponses de query → refilees dans le wiki"
  - Faire des lint passes periodiques (meme manuellement dirigees)

- **A evaluer plus tard** :
  - Creer un `index.md` global (les MOCs thematiques suffisent peut-etre)
  - Creer un `log.md` chronologique (a voir si plus utile que le journal par jour)
  - Creer un dossier `raw/` dedie (depend de si on a besoin de PDFs locaux)

- **A ne pas adopter tel quel** :
  - La discipline rigide "LLM only writes wiki" : dans un projet de memoire, on aura besoin d'annotations perso
  - Un refactor complet de l'arborescence 00-09

## 10. Questions a trancher avant adoption

- [ ] Veut-on telecharger les PDFs localement (+ besoin d'un `raw/`) ou garder les URLs ?
- [ ] Le `log.md` chronologique apporterait-il plus que le journal par jour actuel ?
- [ ] Faut-il marquer explicitement les pages "LLM-owned" vs "human-owned" via frontmatter ?
- [ ] Adopte-t-on le pattern "query output → page du wiki" pour les comparaisons/analyses ?
- [ ] A quelle frequence faire des lint passes ?

## 11. Sources

- Article original : "LLM Wiki" (pattern abstrait, communaute Claude Code / Codex, 2025-2026)
- Inspiration : Andrej Karpathy, "LLM Knowledge Bases" (Twitter/X)
- Filiation historique :
  - Vannevar Bush, "As We May Think" (1945) — Memex
  - Niklas Luhmann — Zettelkasten
  - Tolkien Gateway et autres fan wikis — modele communautaire

## Notes personnelles

- Le pattern est seduisant mais il faut resister a l'impulsion de tout refactorer. Notre structure 00-09 marche deja.
- Les elements les plus immediatement utiles sont : (1) les workflows explicites dans CLAUDE.md, (2) le reflexe "query output → wiki page", (3) les lint passes regulieres.
- Question ouverte : est-ce qu'on utilise deja implicitement ce pattern ? Oui, partiellement — chaque fois qu'on a lu un papier et mis a jour la fiche, on a fait de l'ingest. Chaque fois qu'on a cree la synthese GraphRAG, on a fait du "query output filed back".
- A discuter : en fait, nos dernieres actions dans cette session (lecture de Belikov, d'Amato, Zhang + creation de la synthese typologie) correspondent deja a des ingest+query du pattern. Sans l'avoir formalise.
