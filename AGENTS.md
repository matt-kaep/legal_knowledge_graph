# Instructions pour agents IA (Codex, Claude Code, etc.)

Ce vault Obsidian est un projet de recherche sur la construction d'un Knowledge Graph juridique francais. Ce fichier est identique a `CLAUDE.md` (dupliqué pour les agents non-Claude qui lisent `AGENTS.md` par convention).

**Pour reprendre une session existante**, lire en priorité :
- `01-Projet/handoffs/handoff-CODEX-2026-06-09.md` — point d'entrée master (Codex)
- `01-Projet/handoffs/handoff-SESSION-PRINCIPALE-2026-06-09.md` — état des 4 chantiers
- `01-Projet/handoffs/handoff-LightGCN-v2-2026-06-08.md` — détails LightGCN
- `01-Projet/handoffs/handoff-M3-COMPLET.md` — détails M3 LLM-judge

**Conventions techniques transverses** :
- Données + `.venv` vivent UNIQUEMENT dans le checkout principal `/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/` (gitignorés, absents des worktrees)
- Les scripts hardcodent `REPO = Path("/Users/.../legal_knowledge_graph")` (chemin absolu) → lisent/écrivent dans le checkout principal même si exécutés depuis un worktree
- `pdflatex` : `/usr/local/texlive/2025/bin/universal-darwin/pdflatex` (pas dans le PATH)
- Branche active : `etape1-embedding-pur`

## Structure du vault

```
00-Inbox/          -> Notes rapides, captures, a trier
01-Projet/         -> Objectifs, roadmap, journal, decisions
02-Etat-de-l-art/  -> Fiches de lecture par thematique
03-Concepts/       -> Entites, relations, architectures du KG
04-Donnees/        -> Sources FR, APIs, datasets
05-Technique/      -> Stack, prototypes, prompts
06-Analyses/       -> Comparatifs, syntheses, questions ouvertes
07-Redaction/      -> Sections du memoire, biblio, figures
08-Templates/      -> Templates Obsidian
09-Archives/       -> Anciens fichiers, attachments, PDFs
```

## Conventions

### Nommage
- Fichiers : `Nom-En-Kebab-Case.md`
- Tags : `#kebab-case`
- Frontmatter YAML obligatoire sur chaque note

### Tags principaux
- `#article` : fiche de lecture
- `#concept` : definition d'un concept
- `#decision` : ADR (Architecture Decision Record)
- `#question` : question de recherche ouverte
- `#fiche` : fiche de synthese
- `#journal` : entree de journal

### Statuts articles
- `a-lire` -> `en-cours` -> `lu` -> `archive`

### Pertinence
- `haute` : directement applicable au projet
- `moyenne` : contexte utile ou methodologie transposable
- `basse` : culture generale, reference secondaire

## Commandes pour Claude Code

### Creer une fiche de lecture
Utiliser le template `08-Templates/Template-Article.md` et placer dans le bon sous-dossier de `02-Etat-de-l-art/`.

### Creer une synthese
Agreger les fiches d'un dossier et produire un resume dans `06-Analyses/syntheses/`.

### Mettre a jour le journal
Creer une entree datee dans `01-Projet/journal/` avec le template journal.

### Generer la bibliographie
Parser toutes les fiches `#article` et generer un fichier BibTeX dans `07-Redaction/references/`.

## Plugins recommandes
- **Dataview** : requetes dynamiques sur le vault (obligatoire)
- **Templater** : templates avec variables dynamiques
- **Obsidian Git** : versioning du vault
- **Tag Wrangler** : gestion des tags
- **Table Editor** : edition de tableaux markdown
- **Citation Plugin** : gestion bibliographique (BibTeX)
