# doctrine_qgen — découpage hybride à whitelists strictes par section

Date : 2026-05-19
Statut : approuvé (design), implémentation en cours

## Problème

`phase_1` génère 1 prompt par section L1. Sur 19 sections L1 (5 Jurisclasseurs),
4 dépassent le contexte max de Gemma 4 (32768 tok) :

- `app_art_11` L1_001 — ~32k tok texte
- `art_114_a_121` L1_001 — ~42k
- `art_114_a_121` L1_002 — ~31k
- `art_11_secret` L1_002 — ~26k (limite avec whitelist)

Réduire à la maille L2 fait passer tout le monde sous ~28k, mais les
`l2_children` du JSON parsé ne portent QUE `{section_id, title,
offset_start, offset_end}` — ni texte, ni whitelist propre.

## Décision

Découpage **hybride** + whitelists **strictes par unité**, fait dans le
parser (« on fait bien une fois »).

Principe : `phase_0` produit des unités correctement scopées à toutes les
granularités ; `phase_1` choisit celle qui rentre dans le contexte.
phase_0 ne connaît ni modèle ni tokenizer ; phase_1 ne refait aucune
extraction.

## Changements

### phase_0_parse_doctrine_pdfs.py — `group_sections_hierarchical`

Pour chaque `l2_child`, calculer ses propres `articles_in_span` /
`jp_in_span` via le filtre offset DÉJÀ existant (lignes 413-440), appliqué
à `[L2.offset_start, L2.offset_end)` au lieu de `[L1.start, L1.end)`.
Ajouter aussi `text` = `text[L2.offset_start:L2.offset_end]`.
Le L1 conserve ses whitelists agrégées (inchangé). Aucune logique nouvelle :
réutilise `articles_with_offset` / `jp_with_meta` déjà passés en argument.

Conséquence : chaque L2 devient une unité autonome
`{section_id, title, text, offset_start, offset_end, articles_in_span,
jp_in_span}`. Re-parse des 5 PDF requis (les JSON actuels n'ont pas les
whitelists L2).

### phase_1_generate_crfpa_questions_via_vllm.py — boucle de génération

1. Construire une liste plate d'**unités** :
   - estimer la taille du prompt L1 ; si ≤ budget (max_len − max_tokens_out
     − marge) → unité = L1 entier (comportement actuel) ;
   - sinon → unités = ses `l2_children` (autonomes via phase_0).
2. `build_prompt` prend une unité de forme identique L1/L2
   (`title`, `text`, `articles_in_span`, `jp_in_span`).
3. `section_id` / `source_section_offsets` = ceux de l'unité émise.
4. `--max-tokens-out` défaut 2048 → 4096.
5. Skip des unités dont le texte < ~200 tokens (bruit parser : titres
   bibliographiques type « I. — Tricot-Chamard »).
6. Garde-fou `openai.BadRequestError` → marqueur `_error:"vllm_400"`
   conservé (filet, ne devrait plus se déclencher).

## Garanties

- Whitelist stricte par unité : une question issue de `L2_C` ne peut citer
  que les articles/JP textuellement présents dans `L2_C`. Pas de
  décollement du gold. Phase 2 inchangée, toujours stricte.
- 15 L1 entiers : sortie strictement identique à aujourd'hui.
- 4 L1 géants : générés par sous-parties autonomes.
- Pas de duplication de logique d'extraction.

## Tests (local, sans GPU)

- T1 phase_0 : chaque L2 a `text` non vide, `articles_in_span` /
  `jp_in_span` ⊆ ceux du L1 parent (cohérence offset).
- T2 phase_1 : la liste d'unités = 15 L1 + ~14 L2 ; chaque unité produit
  un prompt sous le budget.
- T3 `build_prompt` : OK sur unité L1 et unité L2 (placeholders substitués,
  accolades littérales préservées).
- T4 robustesse : run complet simulé survit (déjà couvert par le test
  `vllm_400`).

## Hors scope

Qualité fine de détection de sections de phase_0 (artefacts
bibliographiques résiduels) — atténuée par le skip < 200 tok, non traitée.

## Trade-off assumé

Re-parsing des 5 PDF (déjà sur le cluster, ou régénéré en local et
drag-and-droppé). Coût CPU négligeable.
