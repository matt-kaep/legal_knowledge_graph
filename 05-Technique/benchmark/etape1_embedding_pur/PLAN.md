# Étape 1 — Embedding pur (articles + JP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** établir le plancher de retrieval question↔article et question↔JP, sans graphe,
avec un encodeur unique BGE-M3 et un linkage `pair_key ↔ texte LEGI ↔ embedding ↔ nœud graphe` aligné par construction.

**Architecture:** quatre scripts CLI numérotés (`01_token_stats.py` … `04_eval_recall.py`)
appellent un package `etape1/` testable par fonction. Tout artefact aligné positionnellement
sur `article_ids` / `jp_ids` du graphe `graph_penal.npz`. Diagnostic de couverture *avant*
embedding, recall@K *sans graphe*.

**Tech Stack:** Python 3.12 · sentence-transformers (BGE-M3) · legi.py (Legilibre) · pyarrow · numpy · scipy.sparse · pytest · Mac MPS (cluster L40S en repli).

**Design doc validé :** `01-Projet/presentations/Plan-Etape1-Embedding-Pur-2026-05-19.html`

---

## File Structure

À créer sous `05-Technique/benchmark/etape1_embedding_pur/` :

```
PLAN.md                            (ce fichier)
README.md                          (run end-to-end + setup système)
requirements.txt
run_all.sh                         (orchestration shell des 4 étapes)
etape1/
  __init__.py
  config.py                        (chemins, constantes, mapping code_slug→LEGI)
  normalize.py                     (pair_key compact ↔ formats LEGI candidates)
  legi_ingest.py                   (wrapper legi.py : download + SQLite build)
  resolve.py                       (pair_key → texte LEGI + couverture)
  linkage.py                       (articles_order, pairkey_to_graphcol, jp_order)
  tokenize_stats.py                (Stage 1 : distribution longueurs)
  embed.py                         (BGE-M3, articles + JP, aligné)
  eval_recall.py                   (recall@K, K*, pourvoi-regex)
scripts/
  01_token_stats.py
  02_fetch_articles.py
  03_embed.py
  04_eval_recall.py
tests/
  __init__.py
  test_normalize.py
  test_resolve.py
  test_linkage.py
  test_eval_recall.py
  fixtures/
    mini_graph.npz                 (graphe synthétique 5 articles × 3 JP)
    mini_legi.sqlite               (3 articles factices)
data/                              (gitignoré, sorties d'exécution)
  articles_penal.parquet
  articles_coverage.json
  articles_order.npy
  pairkey_to_graphcol.npy
  jp_order.npy
  jp_to_graphrow.npy
  emb_articles.npy
  emb_jp.npy
  token_stats.json
  recall_curves.csv
  recall_kstar.json
```

**Inputs read-only (existants, ne pas modifier) :**
- `05-Technique/benchmark/baseline_b2/penal_bundle/graph_penal.npz` (8 085 articles pénaux sur 87 821 colonnes totales)
- `05-Technique/benchmark/baseline_b2/penal_bundle/jp_index_penal.parquet` (118 112 JP, champs `id`, `number`, `juris`, `text`, `summary`)
- `05-Technique/benchmark/baseline_b2/penal_bundle/rubrics_penal.json` (8 questions CRFPA)

---

## Pinned decisions

| Décision | Valeur retenue |
|---|---|
| Modèle | BGE-M3 (`BAAI/bge-m3`, dim 1024, contexte 8 192 tokens), L2-normalisé |
| Périmètre | 4 codes pénaux (8 085 articles nœuds du graphe) |
| Hardware | Mac MPS d'abord (cluster L40S en repli — flag `--device cuda`) |
| Politique troncature | Conditionnelle : décidée après Stage 1 selon p100 mesuré |
| K* article-side | min K tel que recall@K(`articles_attendus.obligatoires`) ≥ 0,5 |
| K* JP-side | min K tel que recall@K(`jp_attendues` résolues par regex pourvoi) ≥ 0,5 |
| Métrique secondaire | recall@K pondéré : `obligatoires` poids 1, `optionnels` poids 0,5 |

---

## Task 1 : Scaffold + config + smoke pytest

**Files:**
- Create: `etape1_embedding_pur/requirements.txt`
- Create: `etape1_embedding_pur/etape1/__init__.py` (vide)
- Create: `etape1_embedding_pur/etape1/config.py`
- Create: `etape1_embedding_pur/tests/__init__.py` (vide)
- Create: `etape1_embedding_pur/tests/test_config.py`
- Create: `etape1_embedding_pur/README.md`
- Create: `etape1_embedding_pur/.gitignore`

- [ ] **Step 1.1 : Écrire `requirements.txt`**

```text
numpy>=1.26
pyarrow>=15
scipy>=1.12
pandas>=2.1
sentence-transformers>=3.0
transformers>=4.42
torch>=2.3
tqdm>=4.66
legi>=0.8
pytest>=8.0
```

- [ ] **Step 1.2 : Écrire `etape1/config.py`**

```python
"""Étape 1 — chemins, constantes, mapping code_slug → nom LEGI officiel."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Inputs read-only (relatifs depuis ROOT)
BENCH = ROOT.parents[0]  # 05-Technique/benchmark/
BUNDLE = BENCH / "baseline_b2" / "penal_bundle"
GRAPH_NPZ = BUNDLE / "graph_penal.npz"
JP_INDEX = BUNDLE / "jp_index_penal.parquet"
RUBRICS = BUNDLE / "rubrics_penal.json"

# LEGI SQLite (produit par Task 3)
LEGI_DIR = DATA / "legi"
LEGI_DIR.mkdir(exist_ok=True)
LEGI_SQLITE = LEGI_DIR / "legi.sqlite"

# Sorties
ARTICLES_PARQUET = DATA / "articles_penal.parquet"
ARTICLES_COVERAGE = DATA / "articles_coverage.json"
ARTICLES_ORDER = DATA / "articles_order.npy"
PAIRKEY_TO_GRAPHCOL = DATA / "pairkey_to_graphcol.npy"
JP_ORDER = DATA / "jp_order.npy"
JP_TO_GRAPHROW = DATA / "jp_to_graphrow.npy"
EMB_ARTICLES = DATA / "emb_articles.npy"
EMB_JP = DATA / "emb_jp.npy"
TOKEN_STATS = DATA / "token_stats.json"
RECALL_CURVES = DATA / "recall_curves.csv"
RECALL_KSTAR = DATA / "recall_kstar.json"

# Modèle
MODEL_ID = "BAAI/bge-m3"
EMB_DIM = 1024
MAX_CTX = 8192

# 4 codes pénaux : code_slug → nom officiel LEGI
PENAL_CODES: dict[str, str] = {
    "code_penal":                              "Code pénal",
    "code_de_procedure_penale":                "Code de procédure pénale",
    "code_de_la_route":                        "Code de la route",
    "code_de_la_justice_penale_des_mineurs":   "Code de la justice pénale des mineurs",
}

# Eval
KS = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
KSTAR_THRESHOLD = 0.5
```

- [ ] **Step 1.3 : Écrire `tests/test_config.py`**

```python
from etape1 import config

def test_inputs_exist():
    assert config.GRAPH_NPZ.exists(), config.GRAPH_NPZ
    assert config.JP_INDEX.exists(), config.JP_INDEX
    assert config.RUBRICS.exists(), config.RUBRICS

def test_penal_codes():
    assert len(config.PENAL_CODES) == 4
    assert "code_penal" in config.PENAL_CODES
    assert config.PENAL_CODES["code_de_procedure_penale"] == "Code de procédure pénale"
```

- [ ] **Step 1.4 : Écrire `README.md` minimal**

````markdown
# Étape 1 — Embedding pur

## Prérequis système
```bash
brew install libarchive hunspell        # macOS, deps de legi.py
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run end-to-end
```bash
./run_all.sh
```

## Étapes individuelles
```bash
python scripts/01_token_stats.py
python scripts/02_fetch_articles.py
python scripts/03_embed.py            # ~ 20 min sur Mac MPS
python scripts/04_eval_recall.py
```

## Cluster L40S (repli)
```bash
python scripts/03_embed.py --device cuda --batch 64
```
````

- [ ] **Step 1.5 : Écrire `.gitignore`**

```text
data/
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 1.6 : Installer + lancer le smoke pytest**

```bash
cd 05-Technique/benchmark/etape1_embedding_pur/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/test_config.py -v
```
Attendu : 2 tests PASS.

- [ ] **Step 1.7 : Commit**

```bash
git add 05-Technique/benchmark/etape1_embedding_pur/
git commit -m "feat(etape1): scaffold + config + smoke test"
```

---

## Task 2 : Module `normalize` — pair_key ↔ candidats LEGI (TDD)

**Files:**
- Create: `etape1/normalize.py`
- Create: `tests/test_normalize.py`

Le format pair_key (cf. `iterate_regex.py`) : `<code_slug>:<article>` où `article` est
compact (préfixe `L|R|D|A|E` collé, suffixes latins en minuscule, lettre finale en majuscule :
`L743-7`, `1649quinquiesB`). LEGI stocke `num` avec espaces et points : `L. 743-7`, `1649 quinquies B`.
La normalisation doit produire **plusieurs candidats** pour matcher les variantes LEGI.

- [ ] **Step 2.1 : Écrire les tests d'abord**

```python
# tests/test_normalize.py
from etape1.normalize import parse_pair_key, legi_num_candidates

def test_parse_simple():
    assert parse_pair_key("code_penal:222-23") == ("code_penal", "222-23")

def test_parse_letter_prefix():
    assert parse_pair_key("code_de_procedure_penale:L743-7") == ("code_de_procedure_penale", "L743-7")

def test_parse_strips_whitespace():
    assert parse_pair_key("  code_penal:222-23  ") == ("code_penal", "222-23")

def test_parse_rejects_malformed():
    import pytest
    with pytest.raises(ValueError):
        parse_pair_key("no_colon_here")

def test_candidates_plain_numeric():
    cands = legi_num_candidates("222-23")
    assert "222-23" in cands

def test_candidates_letter_prefix_expands():
    cands = legi_num_candidates("L743-7")
    # LEGI peut stocker "L. 743-7", "L743-7", "L 743-7"
    assert "L743-7"  in cands
    assert "L. 743-7" in cands
    assert "L 743-7"  in cands

def test_candidates_latin_suffix():
    cands = legi_num_candidates("1649quinquiesB")
    # LEGI : "1649 quinquies B"
    assert "1649 quinquies B" in cands
    assert "1649quinquiesB"   in cands  # forme compacte conservée comme fallback

def test_candidates_bis_lowercase():
    cands = legi_num_candidates("L122-1bis")
    assert "L. 122-1 bis"     in cands
    assert "L122-1bis"        in cands
```

- [ ] **Step 2.2 : Lancer les tests, vérifier qu'ils échouent**

```bash
PYTHONPATH=. pytest tests/test_normalize.py -v
```
Attendu : tous FAIL (`ModuleNotFoundError: etape1.normalize`).

- [ ] **Step 2.3 : Implémenter `etape1/normalize.py`**

```python
"""Conversion pair_key (forme compacte v5) ↔ formats `num` candidats côté LEGI.

Le format pair_key v5 :
  <code_slug>:<article>
  article : préfixe optionnel L|R|D|A|E collé, suffixes latins minuscule,
            lettre finale majuscule (ex. `L743-7`, `1649quinquiesB`, `L122-1bis`).

LEGI peut stocker le num avec :
  - point après la lettre de partie : "L. 743-7"
  - espaces autour des suffixes : "1649 quinquies B"
  - tout collé : "L743-7"
On renvoie tous les candidats raisonnables, le caller essaye dans l'ordre.
"""
from __future__ import annotations
import re

_LATIN_SUFFIXES = ("bis", "ter", "quater", "quinquies", "sexies", "septies",
                   "octies", "novies", "decies", "undecies", "duodecies")
_PART_LETTERS = ("L", "R", "D", "A", "E")

_PAIR_KEY_RE = re.compile(r"^\s*([a-z0-9_]+):([^\s]+)\s*$")


def parse_pair_key(pk: str) -> tuple[str, str]:
    """Parse `<code_slug>:<article_num>`. Lève ValueError si malformé."""
    m = _PAIR_KEY_RE.match(pk)
    if not m:
        raise ValueError(f"pair_key malformé : {pk!r}")
    return m.group(1), m.group(2)


def _split_part_letter(num: str) -> tuple[str | None, str]:
    """`L743-7` → ('L', '743-7'). `222-23` → (None, '222-23')."""
    if num and num[0] in _PART_LETTERS and len(num) > 1 and not num[1].isalpha():
        return num[0], num[1:]
    return None, num


def _split_latin_suffix(rest: str) -> tuple[str, str | None, str | None]:
    """`1649quinquiesB` → ('1649', 'quinquies', 'B'). `122-1bis` → ('122-1', 'bis', None)."""
    for suf in _LATIN_SUFFIXES:
        if suf in rest:
            i = rest.lower().index(suf)
            head = rest[:i]
            after = rest[i + len(suf):]
            tail = after if after else None
            return head, suf, tail
    # Pas de suffixe latin — éventuelle lettre finale isolée
    m = re.match(r"^(.*?)([A-Z])$", rest)
    if m and len(m.group(1)) > 0:
        return m.group(1), None, m.group(2)
    return rest, None, None


def legi_num_candidates(compact: str) -> list[str]:
    """Renvoie une liste ordonnée de variantes plausibles côté LEGI.

    Stratégie : on déconstruit la forme compacte, on régénère avec
    différentes politiques d'espacement et de point.
    """
    part, rest = _split_part_letter(compact)
    head, suf, tail = _split_latin_suffix(rest)

    cands: list[str] = []

    def emit(*pieces: str) -> None:
        s = "".join(p for p in pieces if p)
        if s and s not in cands:
            cands.append(s)

    # 1) forme compacte d'origine
    cands.append(compact)

    # 2) variantes avec point + espace après la lettre de partie
    for part_fmt in ([part, f"{part} ", f"{part}. "] if part else [""]):
        # 3) variantes d'espacement pour le suffixe latin
        suf_variants = []
        if suf:
            suf_variants = [f"{suf}", f" {suf}", f" {suf} "]
        else:
            suf_variants = [""]
        # 4) tail (lettre finale)
        tail_variants = []
        if tail:
            tail_variants = [tail, f" {tail}", f"{tail}"]
        else:
            tail_variants = [""]
        for sv in suf_variants:
            for tv in tail_variants:
                emit(part_fmt, head, sv, tv)

    return cands
```

- [ ] **Step 2.4 : Lancer les tests, vérifier qu'ils passent**

```bash
PYTHONPATH=. pytest tests/test_normalize.py -v
```
Attendu : 7 tests PASS.

- [ ] **Step 2.5 : Commit**

```bash
git add etape1/normalize.py tests/test_normalize.py
git commit -m "feat(etape1): normalize pair_key↔LEGI num avec candidats multiples (TDD)"
```

---

## Task 3 : Ingestion LEGI via `legi.py`

C'est l'étape lourde et orchestrale : télécharger le dump DILA + construire la SQLite.
**Cette tâche n'a pas de TDD** — c'est de l'I/O réseau et le build d'une base. Validation
par requête SQL après build.

**Files:**
- Create: `etape1/legi_ingest.py`
- Create: `scripts/_setup_legi.sh`

- [ ] **Step 3.1 : Écrire `scripts/_setup_legi.sh`**

```bash
#!/usr/bin/env bash
# Télécharge et construit la SQLite LEGI à partir du dump DILA.
# Idempotent : skip si la SQLite existe déjà avec >100k articles.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEGI_DIR="$ROOT/data/legi"
mkdir -p "$LEGI_DIR/fonds"
cd "$LEGI_DIR"

if [ -f legi.sqlite ]; then
  N=$(sqlite3 legi.sqlite 'SELECT COUNT(*) FROM articles;' 2>/dev/null || echo 0)
  if [ "${N:-0}" -gt 100000 ]; then
    echo "✓ legi.sqlite déjà construite ($N articles) — skip"
    exit 0
  fi
fi

echo "→ Téléchargement archives LEGI (échanges DILA, ftp://echanges.dila.gouv.fr/LEGI/)"
echo "   ~3 GB, peut prendre 30-60 min selon réseau"
# Récupérer dernier freemium tar.gz + delta tar.gz
# DILA publie : LEGI_<YYYYMMDD-HHMMSS>.tar.gz (snapshot complet) puis deltas
python -m legi.download fonds/

echo "→ Construction SQLite via legi.tar2sqlite"
python -m legi.tar2sqlite legi.sqlite fonds/

N=$(sqlite3 legi.sqlite 'SELECT COUNT(*) FROM articles;')
echo "✓ legi.sqlite OK ($N articles)"
```

```bash
chmod +x scripts/_setup_legi.sh
```

- [ ] **Step 3.2 : Écrire `etape1/legi_ingest.py`**

```python
"""Wrapper minimal sur la SQLite produite par legi.py.

Schéma legi.py (cf. https://github.com/Legilibre/legi.py) :
  articles(id PRIMARY KEY, section, num, etat, date_debut, date_fin, ...)
  textes_versions(id, titre, ...)
  ... + tables sommaires/structure pour relier articles ↔ code

On expose ici une seule fonction : `load_code_articles(legi_db, code_titre)`
qui renvoie un DataFrame[id, num, texte, etat] pour tous les articles d'un code.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd


def load_code_articles(legi_db: Path, code_titre: str) -> pd.DataFrame:
    """Charge tous les articles 'VIGUEUR' du code `code_titre` (ex. 'Code pénal').

    Retour : DataFrame avec colonnes id, num, texte, etat.
    """
    with sqlite3.connect(legi_db) as cx:
        # legi.py utilise textes_structs pour lier articles à un texte
        q = """
        SELECT a.id, a.num, a.bloc_textuel AS texte, a.etat
        FROM articles a
        JOIN sommaires s ON s.element = a.id
        JOIN textes_versions tv ON tv.id = s.cid_parent
        WHERE tv.titre_court = ? AND a.etat = 'VIGUEUR'
        """
        df = pd.read_sql_query(q, cx, params=(code_titre,))
    return df


def count_articles(legi_db: Path) -> int:
    with sqlite3.connect(legi_db) as cx:
        return cx.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
```

- [ ] **Step 3.3 : Exécuter le setup (long, hors session de codage)**

```bash
cd 05-Technique/benchmark/etape1_embedding_pur/
./scripts/_setup_legi.sh
```
Attendu (durée 30-60 min, dépend du réseau) :
```
✓ legi.sqlite OK (>1 500 000 articles)
```

> ⚠ Si `legi.download` n'existe pas comme module, fallback manuel : `curl -O ftp://echanges.dila.gouv.fr/LEGI/Freemium_legi_global_<date>.tar.gz` dans `fonds/` puis lancer uniquement `python -m legi.tar2sqlite legi.sqlite fonds/`. Documenter le hash de l'archive utilisée dans `data/legi/SOURCE.txt`.

- [ ] **Step 3.4 : Vérifier la SQLite avec une requête sanity**

```bash
sqlite3 data/legi/legi.sqlite "SELECT num, substr(bloc_textuel,1,80) FROM articles WHERE num='222-23' LIMIT 3;"
```
Attendu : ≥1 ligne contenant un extrait commençant par "Tout acte de pénétration sexuelle…".

- [ ] **Step 3.5 : Commit (script + wrapper, PAS la base)**

```bash
git add etape1/legi_ingest.py scripts/_setup_legi.sh
git commit -m "feat(etape1): legi.py ingestion wrapper + setup script"
```

---

## Task 4 : Résolution `pair_key` → texte + diagnostic de couverture

**Files:**
- Create: `etape1/resolve.py`
- Create: `tests/test_resolve.py`
- Create: `tests/fixtures/mini_legi.sqlite` (créée programmatiquement par le test)

- [ ] **Step 4.1 : Écrire les tests d'abord**

```python
# tests/test_resolve.py
import sqlite3
import pandas as pd
import pytest
from etape1.resolve import resolve_pair_keys, coverage_report

@pytest.fixture
def mini_db(tmp_path):
    """Mini SQLite avec 3 articles factices, schéma proche de legi.py."""
    db = tmp_path / "mini.sqlite"
    with sqlite3.connect(db) as cx:
        cx.executescript("""
        CREATE TABLE articles(id TEXT PRIMARY KEY, num TEXT,
                              bloc_textuel TEXT, etat TEXT);
        CREATE TABLE sommaires(element TEXT, cid_parent TEXT);
        CREATE TABLE textes_versions(id TEXT, titre_court TEXT);
        INSERT INTO textes_versions VALUES ('T1','Code pénal');
        INSERT INTO articles VALUES ('A1','222-23','Acte de pénétration…','VIGUEUR');
        INSERT INTO articles VALUES ('A2','L. 743-7','Procédure spéciale…','VIGUEUR');
        INSERT INTO articles VALUES ('A3','1649 quinquies B','Disposition fiscale…','VIGUEUR');
        INSERT INTO sommaires VALUES ('A1','T1');
        INSERT INTO sommaires VALUES ('A2','T1');
        INSERT INTO sommaires VALUES ('A3','T1');
        """)
    return db

def test_resolve_plain(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:222-23"], {"code_penal": "Code pénal"})
    assert out["code_penal:222-23"]["texte"].startswith("Acte de pénétration")
    assert out["code_penal:222-23"]["matched_num"] == "222-23"

def test_resolve_letter_prefix_variant(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:L743-7"], {"code_penal": "Code pénal"})
    assert out["code_penal:L743-7"]["texte"].startswith("Procédure")
    assert out["code_penal:L743-7"]["matched_num"] == "L. 743-7"

def test_resolve_latin_suffix(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:1649quinquiesB"], {"code_penal": "Code pénal"})
    assert out["code_penal:1649quinquiesB"]["texte"].startswith("Disposition")

def test_resolve_missing(mini_db):
    out = resolve_pair_keys(mini_db, ["code_penal:999-99"], {"code_penal": "Code pénal"})
    assert out["code_penal:999-99"]["texte"] is None
    assert out["code_penal:999-99"]["matched_num"] is None

def test_coverage_report_counts(mini_db):
    pks = ["code_penal:222-23", "code_penal:L743-7", "code_penal:999-99"]
    gold = {"code_penal:222-23"}  # 1 gold présent
    out = resolve_pair_keys(mini_db, pks, {"code_penal": "Code pénal"})
    rep = coverage_report(out, gold_pair_keys=gold)
    assert rep["n_total"] == 3
    assert rep["n_resolved"] == 2
    assert rep["resolution_rate"] == pytest.approx(2/3, rel=1e-3)
    assert rep["n_gold_total"] == 1
    assert rep["n_gold_resolved"] == 1
    assert rep["gold_resolution_rate"] == 1.0
```

- [ ] **Step 4.2 : Lancer les tests, vérifier qu'ils échouent**

```bash
PYTHONPATH=. pytest tests/test_resolve.py -v
```
Attendu : tous FAIL.

- [ ] **Step 4.3 : Implémenter `etape1/resolve.py`**

```python
"""Résolution pair_key → texte LEGI + diagnostic de couverture.

Stratégie : pour chaque pair_key, tente chaque candidat de `legi_num_candidates`
contre la SQLite, premier match gagnant. Renvoie un dict structuré exploitable
pour produire articles_penal.parquet et articles_coverage.json.
"""
from __future__ import annotations
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypedDict
from .normalize import parse_pair_key, legi_num_candidates


class ResolveEntry(TypedDict):
    code_slug: str
    code_titre: str
    num_compact: str
    matched_num: str | None
    texte: str | None


def resolve_pair_keys(
    legi_db: Path,
    pair_keys: Iterable[str],
    code_map: Mapping[str, str],
) -> dict[str, ResolveEntry]:
    """Renvoie {pair_key: ResolveEntry} pour chaque pair_key fourni."""
    out: dict[str, ResolveEntry] = {}
    sql = """
    SELECT a.num, a.bloc_textuel
    FROM articles a
    JOIN sommaires s ON s.element = a.id
    JOIN textes_versions tv ON tv.id = s.cid_parent
    WHERE tv.titre_court = ? AND a.num = ? AND a.etat = 'VIGUEUR'
    LIMIT 1
    """
    with sqlite3.connect(legi_db) as cx:
        for pk in pair_keys:
            slug, compact = parse_pair_key(pk)
            titre = code_map.get(slug)
            entry: ResolveEntry = {
                "code_slug":   slug,
                "code_titre":  titre or "",
                "num_compact": compact,
                "matched_num": None,
                "texte":       None,
            }
            if titre is None:
                out[pk] = entry
                continue
            for cand in legi_num_candidates(compact):
                row = cx.execute(sql, (titre, cand)).fetchone()
                if row is not None:
                    entry["matched_num"] = row[0]
                    entry["texte"] = row[1]
                    break
            out[pk] = entry
    return out


def coverage_report(
    resolved: dict[str, ResolveEntry],
    gold_pair_keys: set[str],
) -> dict:
    """Compte global et sur le sous-ensemble gold."""
    n_total = len(resolved)
    n_resolved = sum(1 for e in resolved.values() if e["texte"] is not None)
    gold_in_set = gold_pair_keys & set(resolved.keys())
    n_gold_total = len(gold_in_set)
    n_gold_resolved = sum(1 for pk in gold_in_set if resolved[pk]["texte"] is not None)
    return {
        "n_total":              n_total,
        "n_resolved":           n_resolved,
        "resolution_rate":      n_resolved / max(n_total, 1),
        "n_gold_total":         n_gold_total,
        "n_gold_resolved":      n_gold_resolved,
        "gold_resolution_rate": n_gold_resolved / max(n_gold_total, 1),
        "missed_examples":      [pk for pk, e in resolved.items() if e["texte"] is None][:20],
    }
```

- [ ] **Step 4.4 : Lancer les tests, vérifier qu'ils passent**

```bash
PYTHONPATH=. pytest tests/test_resolve.py -v
```
Attendu : 5 PASS.

- [ ] **Step 4.5 : Écrire `scripts/02_fetch_articles.py`**

```python
"""CLI : résout les 8085 pair_keys pénaux du graphe → articles_penal.parquet."""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from etape1 import config
from etape1.resolve import resolve_pair_keys, coverage_report


def main() -> int:
    if not config.LEGI_SQLITE.exists():
        print(f"✗ {config.LEGI_SQLITE} absent — lancer ./scripts/_setup_legi.sh d'abord")
        return 1

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    article_ids = z["article_ids"]
    article_codes = z["article_codes"]
    penal_mask = np.array([c in config.PENAL_CODES for c in article_codes])
    penal_pks = article_ids[penal_mask].tolist()
    print(f"Articles pénaux du graphe : {len(penal_pks)}")

    rubrics = json.loads(config.RUBRICS.read_text())["questions"]
    gold = set()
    for q in rubrics:
        aa = q["articles_attendus"]
        gold |= set(aa.get("obligatoires", [])) | set(aa.get("optionnels", []))

    print(f"Résolution via {config.LEGI_SQLITE}…")
    resolved = resolve_pair_keys(config.LEGI_SQLITE, penal_pks, config.PENAL_CODES)

    rep = coverage_report(resolved, gold_pair_keys=gold)
    config.ARTICLES_COVERAGE.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"  Couverture globale : {rep['n_resolved']}/{rep['n_total']} "
          f"({100*rep['resolution_rate']:.1f}%)")
    print(f"  Couverture gold    : {rep['n_gold_resolved']}/{rep['n_gold_total']} "
          f"({100*rep['gold_resolution_rate']:.1f}%)")

    rows = [
        {"pair_key": pk, **resolved[pk]}
        for pk in penal_pks if resolved[pk]["texte"] is not None
    ]
    pd.DataFrame(rows).to_parquet(config.ARTICLES_PARQUET, index=False)
    print(f"✓ {len(rows)} articles écrits dans {config.ARTICLES_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.6 : Lancer le script réel**

```bash
PYTHONPATH=. python scripts/02_fetch_articles.py
```
Attendu (durée ~1 min) :
```
Articles pénaux du graphe : 8085
  Couverture globale : 7xxx/8085 (~9X%)
  Couverture gold    : 5x/51 (98.0%)
✓ 7xxx articles écrits dans data/articles_penal.parquet
```
> Si le taux global tombe sous 85 %, ouvrir `articles_coverage.json["missed_examples"]` et étendre `normalize.legi_num_candidates` ou enrichir `PENAL_CODES` avant de poursuivre. Documenter dans `articles_coverage.json["notes"]`.

- [ ] **Step 4.7 : Commit (sans les données)**

```bash
git add etape1/resolve.py tests/test_resolve.py scripts/02_fetch_articles.py
git commit -m "feat(etape1): pair_key→LEGI text + diagnostic couverture (TDD)"
```

---

## Task 5 : Diagnostic longueurs (Stage 1)

**Files:**
- Create: `etape1/tokenize_stats.py`
- Create: `tests/test_tokenize_stats.py`
- Create: `scripts/01_token_stats.py`

- [ ] **Step 5.1 : Écrire le test**

```python
# tests/test_tokenize_stats.py
from etape1.tokenize_stats import compute_token_stats

class FakeTokenizer:
    """Tokenizer factice : 1 token par caractère."""
    def __call__(self, texts, **kw):
        return {"input_ids": [list(range(len(t))) for t in texts]}

def test_basic_stats():
    texts = ["a", "ab", "abc", "abcd"]  # longueurs 1,2,3,4
    s = compute_token_stats(texts, FakeTokenizer(), max_ctx=3)
    assert s["n"] == 4
    assert s["p50"] in (2, 3)  # tolérance médiane paire
    assert s["p100"] == 4
    assert s["n_over_ctx"] == 1   # seul "abcd" dépasse 3
    assert s["pct_over_ctx"] == 0.25
```

- [ ] **Step 5.2 : Vérifier l'échec**

```bash
PYTHONPATH=. pytest tests/test_tokenize_stats.py -v
```
Attendu : FAIL.

- [ ] **Step 5.3 : Implémenter `etape1/tokenize_stats.py`**

```python
"""Distribution de longueur en tokens d'un corpus."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


def compute_token_stats(texts: Sequence[str], tokenizer, max_ctx: int) -> dict:
    """Tokenise (batché) et renvoie p50/p90/p99/p100 + dépassements `max_ctx`."""
    BATCH = 512
    lengths: list[int] = []
    for i in range(0, len(texts), BATCH):
        batch = list(texts[i : i + BATCH])
        enc = tokenizer(batch, add_special_tokens=False, truncation=False,
                        return_attention_mask=False, return_token_type_ids=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    arr = np.array(lengths, dtype=np.int32)
    over = int((arr > max_ctx).sum())
    return {
        "n":            int(arr.size),
        "p50":          int(np.percentile(arr, 50)),
        "p90":          int(np.percentile(arr, 90)),
        "p99":          int(np.percentile(arr, 99)),
        "p100":         int(arr.max()) if arr.size else 0,
        "mean":         float(arr.mean()) if arr.size else 0.0,
        "max_ctx":      int(max_ctx),
        "n_over_ctx":   over,
        "pct_over_ctx": over / max(arr.size, 1),
    }
```

- [ ] **Step 5.4 : Vérifier le succès**

```bash
PYTHONPATH=. pytest tests/test_tokenize_stats.py -v
```
Attendu : PASS.

- [ ] **Step 5.5 : Écrire `scripts/01_token_stats.py`**

```python
"""CLI : distribution de longueur articles + JP summaries (sous-corpus pénal)."""
from __future__ import annotations
import json
import sys
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from etape1 import config
from etape1.tokenize_stats import compute_token_stats


def main() -> int:
    if not config.ARTICLES_PARQUET.exists():
        print(f"✗ {config.ARTICLES_PARQUET} absent — lancer 02_fetch_articles.py d'abord")
        return 1

    print(f"Chargement tokenizer {config.MODEL_ID}…")
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID)

    arts = pd.read_parquet(config.ARTICLES_PARQUET)
    print(f"Articles : {len(arts)}")
    s_arts = compute_token_stats(arts["texte"].tolist(), tok, max_ctx=config.MAX_CTX)
    print(f"  articles  p50={s_arts['p50']} p90={s_arts['p90']} "
          f"p99={s_arts['p99']} p100={s_arts['p100']} "
          f"over={s_arts['n_over_ctx']} ({100*s_arts['pct_over_ctx']:.2f}%)")

    jp = pq.read_table(config.JP_INDEX, columns=["id", "juris", "summary"]).to_pandas()
    jp = jp.dropna(subset=["summary"])
    jp = jp[jp["summary"].str.len() > 0]
    print(f"JP avec summary : {len(jp)}")
    s_jp = compute_token_stats(jp["summary"].tolist(), tok, max_ctx=config.MAX_CTX)
    print(f"  jp_summary p50={s_jp['p50']} p90={s_jp['p90']} "
          f"p99={s_jp['p99']} p100={s_jp['p100']} "
          f"over={s_jp['n_over_ctx']} ({100*s_jp['pct_over_ctx']:.2f}%)")

    payload = {"articles": s_arts, "jp_summary": s_jp}
    if max(s_arts["n_over_ctx"], s_jp["n_over_ctx"]) == 0:
        payload["truncation_policy"] = "none"
        payload["note"] = "p100 < max_ctx pour les deux corpus → embedding direct sans troncature."
    else:
        payload["truncation_policy"] = "chunk_meanpool_overflow_only"
        payload["note"] = "Au moins un corpus dépasse max_ctx — chunk+mean-pool appliqué aux seuls dépassements (cf. embed.py)."
    config.TOKEN_STATS.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"✓ {config.TOKEN_STATS} écrit (policy: {payload['truncation_policy']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5.6 : Lancer le script (Stage 1 réel)**

```bash
PYTHONPATH=. python scripts/01_token_stats.py
```
Attendu (durée ~2 min, premier appel télécharge BGE-M3 ~2.3 GB) :
```
Articles : 7xxx
  articles  p50=... p100=... over=... (.. %)
JP avec summary : ~93000
  jp_summary p50=... p100=... over=0 (0.00%)
✓ data/token_stats.json (policy: ...)
```

- [ ] **Step 5.7 : Commit**

```bash
git add etape1/tokenize_stats.py tests/test_tokenize_stats.py scripts/01_token_stats.py
git commit -m "feat(etape1): diagnostic longueurs (Stage 1) avec décision tronc. conditionnelle"
```

---

## Task 6 : Linkage artifacts (alignement vers le graphe)

**Files:**
- Create: `etape1/linkage.py`
- Create: `tests/test_linkage.py`

L'invariant central du design. Produit deux paires d'artefacts symétriques (articles / JP)
qui rendent l'alignement vers le graphe **lossless et positionnel**.

- [ ] **Step 6.1 : Écrire les tests**

```python
# tests/test_linkage.py
import numpy as np
import pandas as pd
import pytest
from etape1.linkage import build_articles_linkage, build_jp_linkage

def test_articles_linkage_basic():
    article_ids = np.array(["code_civil:1240", "code_penal:222-23",
                            "code_penal:121-3", "cgi:1559"], dtype=object)
    article_codes = np.array(["code_civil", "code_penal", "code_penal", "cgi"], dtype=object)
    resolved_pks = {"code_penal:222-23", "code_penal:121-3"}  # 2/2 résolus
    penal = {"code_penal"}

    order, p2col = build_articles_linkage(article_ids, article_codes, resolved_pks, penal)
    assert order.tolist() == ["code_penal:222-23", "code_penal:121-3"]
    assert p2col.tolist() == [1, 2]                       # idx dans article_ids
    assert (article_ids[p2col] == order).all()            # round-trip

def test_articles_linkage_skips_unresolved():
    article_ids = np.array(["code_penal:222-23", "code_penal:999-99"], dtype=object)
    article_codes = np.array(["code_penal", "code_penal"], dtype=object)
    resolved = {"code_penal:222-23"}
    order, p2col = build_articles_linkage(article_ids, article_codes, resolved, {"code_penal"})
    assert order.tolist() == ["code_penal:222-23"]
    assert p2col.tolist() == [0]

def test_jp_linkage_filters_no_summary():
    jp_ids = np.array(["a", "b", "c", "d"], dtype=object)
    df = pd.DataFrame({"id": ["a", "b", "c", "d"],
                       "summary": ["x", None, "y", ""]})
    order, j2row = build_jp_linkage(jp_ids, df)
    assert order.tolist() == ["a", "c"]
    assert j2row.tolist() == [0, 2]
```

- [ ] **Step 6.2 : Vérifier l'échec**

```bash
PYTHONPATH=. pytest tests/test_linkage.py -v
```
Attendu : FAIL.

- [ ] **Step 6.3 : Implémenter `etape1/linkage.py`**

```python
"""Artefacts d'alignement embedding-rows ↔ nœuds graphe.

Convention :
  emb_articles.npy[i] est l'embedding du nœud article de colonne
  pairkey_to_graphcol[i] dans le graphe ; articles_order[i] = article_ids[pairkey_to_graphcol[i]].

Idem pour les JP : jp_to_graphrow / jp_order.
"""
from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd


def build_articles_linkage(
    article_ids: np.ndarray,
    article_codes: np.ndarray,
    resolved_pair_keys: Iterable[str],
    penal_codes: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Filtre les colonnes du graphe pour ne garder que les pair_keys :
       (a) appartenant aux codes pénaux, (b) résolus en texte LEGI.
    Préserve l'ordre original de `article_ids`.

    Retour : (articles_order, pairkey_to_graphcol) tous deux len = N_embeddable.
    """
    penal_set = set(penal_codes)
    resolved = set(resolved_pair_keys)
    cols_keep = [i for i, (pk, c) in enumerate(zip(article_ids, article_codes))
                 if c in penal_set and pk in resolved]
    p2col = np.array(cols_keep, dtype=np.int32)
    order = article_ids[p2col].astype(object)
    return order, p2col


def build_jp_linkage(
    jp_ids: np.ndarray,
    jp_index_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtre les lignes JP du graphe pour ne garder que celles avec summary non-vide.
    Préserve l'ordre original de `jp_ids`.

    Retour : (jp_order, jp_to_graphrow) tous deux len = N_jp_with_summary.
    """
    has_sum = {row.id for row in jp_index_df.itertuples()
               if isinstance(row.summary, str) and row.summary.strip()}
    rows_keep = [i for i, jpid in enumerate(jp_ids) if jpid in has_sum]
    j2row = np.array(rows_keep, dtype=np.int32)
    order = jp_ids[j2row].astype(object)
    return order, j2row
```

- [ ] **Step 6.4 : Vérifier le succès**

```bash
PYTHONPATH=. pytest tests/test_linkage.py -v
```
Attendu : 3 PASS.

- [ ] **Step 6.5 : Commit**

```bash
git add etape1/linkage.py tests/test_linkage.py
git commit -m "feat(etape1): linkage artifacts — alignement positionnel lossless (TDD)"
```

---

## Task 7 : Embedding BGE-M3 (articles + JP)

**Files:**
- Create: `etape1/embed.py`
- Create: `scripts/03_embed.py`

- [ ] **Step 7.1 : Écrire `etape1/embed.py`**

```python
"""Embedding BGE-M3, sans troncature (sauf overflow → chunk+mean-pool).

L2-normalisé. Output : (N, EMB_DIM) float32. Aligné sur l'ordre passé en entrée.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from . import config


def _detect_device(override: str | None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _chunk_and_mean(model: SentenceTransformer, tokenizer, text: str,
                    max_ctx: int, batch: int) -> np.ndarray:
    """Pour un texte > max_ctx : tokens → chunks contigus de max_ctx → mean-pool L2."""
    ids = tokenizer(text, add_special_tokens=False, truncation=False,
                    return_attention_mask=False)["input_ids"]
    chunks = [ids[i : i + max_ctx] for i in range(0, len(ids), max_ctx)]
    decoded = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
    embs = model.encode(decoded, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    pooled = embs.mean(axis=0)
    n = np.linalg.norm(pooled)
    return (pooled / n).astype(np.float32) if n > 0 else pooled


def embed_corpus(texts: Sequence[str], device: str | None = None,
                 batch: int = 32) -> np.ndarray:
    """Renvoie (len(texts), EMB_DIM) float32 L2-normalisé, dans l'ordre d'entrée."""
    dev = _detect_device(device)
    print(f"  device={dev}, batch={batch}")
    model = SentenceTransformer(config.MODEL_ID, device=dev)
    model.max_seq_length = config.MAX_CTX
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    # Repérage des dépassements une fois pour toutes
    over_idx: list[int] = []
    SCAN_BATCH = 256
    for i in range(0, len(texts), SCAN_BATCH):
        enc = tokenizer(list(texts[i : i + SCAN_BATCH]),
                        add_special_tokens=False, truncation=False,
                        return_attention_mask=False)["input_ids"]
        for j, ids in enumerate(enc):
            if len(ids) > config.MAX_CTX:
                over_idx.append(i + j)
    if over_idx:
        print(f"  {len(over_idx)} textes > {config.MAX_CTX} tokens → chunk+mean-pool sur ceux-là")

    out = np.zeros((len(texts), config.EMB_DIM), dtype=np.float32)
    over_set = set(over_idx)
    # Encodage normal pour les autres
    keep_idx = [i for i in range(len(texts)) if i not in over_set]
    keep_texts = [texts[i] for i in keep_idx]
    embs = model.encode(keep_texts, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    for k, i in enumerate(keep_idx):
        out[i] = embs[k]
    # Encodage chunké pour les dépassements
    for i in over_idx:
        out[i] = _chunk_and_mean(model, tokenizer, texts[i],
                                  max_ctx=config.MAX_CTX, batch=batch)
    return out
```

- [ ] **Step 7.2 : Écrire `scripts/03_embed.py`**

```python
"""CLI : produit emb_articles.npy + emb_jp.npy + artefacts de linkage."""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from etape1 import config
from etape1.linkage import build_articles_linkage, build_jp_linkage
from etape1.embed import embed_corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="Run sur 10 articles + 10 JP pour valider le pipeline")
    args = ap.parse_args()

    # 1. Linkage articles
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    arts_df = pd.read_parquet(config.ARTICLES_PARQUET)
    resolved_pks = set(arts_df["pair_key"])
    art_order, p2col = build_articles_linkage(
        z["article_ids"], z["article_codes"], resolved_pks, set(config.PENAL_CODES.keys()))
    np.save(config.ARTICLES_ORDER, art_order)
    np.save(config.PAIRKEY_TO_GRAPHCOL, p2col)
    print(f"Articles à embedder : {len(art_order)}")

    # Texte des articles dans l'ordre de art_order
    text_by_pk = dict(zip(arts_df["pair_key"], arts_df["texte"]))
    art_texts = [text_by_pk[pk] for pk in art_order]

    # 2. Linkage JP
    jp_df = pq.read_table(config.JP_INDEX, columns=["id", "juris", "summary"]).to_pandas()
    jp_order, j2row = build_jp_linkage(z["jp_ids"], jp_df)
    np.save(config.JP_ORDER, jp_order)
    np.save(config.JP_TO_GRAPHROW, j2row)
    print(f"JP à embedder       : {len(jp_order)}")

    sum_by_id = dict(zip(jp_df["id"], jp_df["summary"]))
    jp_texts = [sum_by_id[jpid] for jpid in jp_order]

    if args.smoke:
        art_texts, art_order_s, p2col_s = art_texts[:10], art_order[:10], p2col[:10]
        jp_texts,  jp_order_s,  j2row_s = jp_texts[:10],  jp_order[:10],  j2row[:10]
        print("→ SMOKE MODE 10+10")

    # 3. Embedding
    print(f"Embedding articles ({len(art_texts)})…")
    emb_arts = embed_corpus(art_texts, device=args.device, batch=args.batch)
    assert emb_arts.shape == (len(art_texts), config.EMB_DIM), emb_arts.shape
    assert not np.isnan(emb_arts).any(), "NaN dans emb_articles"
    np.save(config.EMB_ARTICLES if not args.smoke else config.DATA / "emb_articles.smoke.npy", emb_arts)

    print(f"Embedding JP ({len(jp_texts)})…")
    emb_jp = embed_corpus(jp_texts, device=args.device, batch=args.batch)
    assert emb_jp.shape == (len(jp_texts), config.EMB_DIM), emb_jp.shape
    assert not np.isnan(emb_jp).any(), "NaN dans emb_jp"
    np.save(config.EMB_JP if not args.smoke else config.DATA / "emb_jp.smoke.npy", emb_jp)

    print(f"✓ Embedding terminé. Aligned :")
    print(f"   emb_articles.npy   ⟂   articles_order.npy   ⟂   article_ids[pairkey_to_graphcol]")
    print(f"   emb_jp.npy         ⟂   jp_order.npy         ⟂   jp_ids[jp_to_graphrow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.3 : Smoke test (10+10) avant le run complet**

```bash
PYTHONPATH=. python scripts/03_embed.py --smoke
```
Attendu (durée ~30 s) :
```
Articles à embedder : 7xxx
JP à embedder       : ~93000
→ SMOKE MODE 10+10
Embedding articles (10)…
  device=mps, batch=32
Embedding JP (10)…
✓ Embedding terminé. Aligned : …
```

- [ ] **Step 7.4 : Vérifier l'alignement avec un check Python**

```bash
PYTHONPATH=. python -c "
import numpy as np
from etape1 import config
z = np.load(config.GRAPH_NPZ, allow_pickle=True)
order = np.load(config.ARTICLES_ORDER, allow_pickle=True)
p2col = np.load(config.PAIRKEY_TO_GRAPHCOL)
assert (z['article_ids'][p2col] == order).all(), 'misaligned'
assert (np.diff(p2col) > 0).all(), 'not strictly increasing'
emb = np.load(config.DATA / 'emb_articles.smoke.npy')
assert emb.shape == (10, 1024), emb.shape
assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3), 'not L2 normalized'
print('✓ alignement + L2-norm OK')
"
```
Attendu : `✓ alignement + L2-norm OK`.

- [ ] **Step 7.5 : Run complet**

```bash
PYTHONPATH=. python scripts/03_embed.py --device mps --batch 32
```
Attendu (durée ~15-25 min sur M-series MPS, ~5-10 min sur L40S) :
```
Articles à embedder : 7xxx
JP à embedder       : ~93000
Embedding articles (7xxx)…
  device=mps, batch=32
Embedding JP (~93000)…
✓ Embedding terminé.
```

- [ ] **Step 7.6 : Commit (code seul, pas les .npy)**

```bash
git add etape1/embed.py scripts/03_embed.py
git commit -m "feat(etape1): embedding BGE-M3 articles+JP aligné sur le graphe"
```

---

## Task 8 : Éval recall top-K (Stage 4)

**Files:**
- Create: `etape1/eval_recall.py`
- Create: `tests/test_eval_recall.py`
- Create: `scripts/04_eval_recall.py`

- [ ] **Step 8.1 : Écrire les tests**

```python
# tests/test_eval_recall.py
import numpy as np
from etape1.eval_recall import (
    recall_at_k, kstar, extract_pourvoi_numbers,
)

def test_recall_at_k_basic():
    ranked = ["a", "b", "c", "d", "e"]
    gold = {"b", "e"}
    assert recall_at_k(ranked, gold, k=1) == 0.0
    assert recall_at_k(ranked, gold, k=2) == 0.5    # 'b' inclus
    assert recall_at_k(ranked, gold, k=5) == 1.0

def test_recall_at_k_no_gold():
    assert recall_at_k(["a", "b"], set(), k=2) == 0.0

def test_kstar_threshold():
    # gold = 2 éléments, threshold 0.5 → besoin de 1 match
    ranked = ["x", "y", "a"]  # 'a' au rang 3
    gold = {"a", "b"}
    assert kstar(ranked, gold, ks=[1, 2, 3, 5], threshold=0.5) == 3

def test_kstar_never_reached():
    ranked = ["x", "y"]
    gold = {"a"}
    assert kstar(ranked, gold, ks=[1, 2], threshold=0.5) is None  # jamais atteint

def test_extract_pourvoi_cc_format():
    text = "Cass. crim., 9 janv. 2019, n 18-82.829"
    assert extract_pourvoi_numbers(text) == ["18-82.829"]

def test_extract_pourvoi_multiple_formats():
    text = "Cass. crim., n° 20-80.135 et n. 90-83.786"
    nums = extract_pourvoi_numbers(text)
    assert "20-80.135" in nums
    assert "90-83.786" in nums
```

- [ ] **Step 8.2 : Vérifier l'échec**

```bash
PYTHONPATH=. pytest tests/test_eval_recall.py -v
```
Attendu : FAIL.

- [ ] **Step 8.3 : Implémenter `etape1/eval_recall.py`**

```python
"""Recall@K, K*, et résolution short_ref → pourvoi pour le côté JP."""
from __future__ import annotations
from collections.abc import Sequence
import re
from typing import Iterable

# Format CC : 18-82.829, 90-83.786 (XX-XX.XXX)
_POURVOI_CC = re.compile(r"\b(\d{2}-\d{2}\.\d{3})\b")


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in gold)
    return hit / len(gold)


def kstar(ranked: Sequence[str], gold: set[str],
          ks: Iterable[int], threshold: float) -> int | None:
    """Plus petit k ∈ ks tel que recall@k ≥ threshold, ou None."""
    for k in sorted(ks):
        if recall_at_k(ranked, gold, k) >= threshold:
            return k
    return None


def extract_pourvoi_numbers(text: str) -> list[str]:
    """Format CC uniquement (cf. journal 2026-05-05, JP-side fragilité connue)."""
    return _POURVOI_CC.findall(text or "")
```

- [ ] **Step 8.4 : Vérifier le succès**

```bash
PYTHONPATH=. pytest tests/test_eval_recall.py -v
```
Attendu : 6 PASS.

- [ ] **Step 8.5 : Écrire `scripts/04_eval_recall.py`**

```python
"""CLI : recall@K question↔article + question↔JP, courbes + K*."""
from __future__ import annotations
import json
import re
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from etape1 import config
from etape1.eval_recall import recall_at_k, kstar, extract_pourvoi_numbers

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")


def _encode_questions(questions: list[str]) -> np.ndarray:
    model = SentenceTransformer(config.MODEL_ID)
    embs = model.encode(questions, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    return embs


def _build_pourvoi_to_jpid_map() -> dict[str, list[str]]:
    """Construit pourvoi (XX-XX.XXX) → [jp_id, ...] depuis jp_index_penal.parquet,
    restreint aux JP CC (les seules où le `number` suit le format)."""
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for row in jp.itertuples():
        n = (row.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(row.id)
    return out


def main() -> int:
    # Charger embeddings + linkage
    emb_arts = np.load(config.EMB_ARTICLES)
    emb_jp = np.load(config.EMB_JP)
    art_order = np.load(config.ARTICLES_ORDER, allow_pickle=True)
    jp_order = np.load(config.JP_ORDER, allow_pickle=True)

    rubrics = json.loads(config.RUBRICS.read_text())["questions"]
    q_texts = [q["question"] for q in rubrics]
    print(f"Encodage de {len(q_texts)} questions…")
    Q = _encode_questions(q_texts)  # (n_q, dim)

    # Similarités (cosinus = produit scalaire car L2-normalisé)
    sim_arts = Q @ emb_arts.T  # (n_q, n_arts)
    sim_jp = Q @ emb_jp.T      # (n_q, n_jp)

    # Mapping pourvoi → liste de jp_ids du corpus pénal CC
    print("Construction map pourvoi → jp_id…")
    pourvoi_to_jpid = _build_pourvoi_to_jpid_map()
    print(f"  {len(pourvoi_to_jpid)} pourvois CC indexables")

    rows = []
    kstar_summary = []
    for qi, q in enumerate(rubrics):
        qid = q["id"]

        # === ARTICLE-SIDE ===
        oblig = set(q["articles_attendus"].get("obligatoires", []))
        ranked_arts = list(art_order[np.argsort(-sim_arts[qi])])
        for k in config.KS:
            rows.append({"question_id": qid, "side": "article",
                          "metric": "obligatoires", "k": k,
                          "recall": recall_at_k(ranked_arts, oblig, k)})
        kstar_a = kstar(ranked_arts, oblig, config.KS, config.KSTAR_THRESHOLD)

        # === JP-SIDE ===
        jp_gold_pourvois = set()
        for jp in q["jp_attendues"]:
            short = jp.get("short_ref") or ""
            jp_gold_pourvois.update(extract_pourvoi_numbers(short))
        jp_gold_ids = {jpid for p in jp_gold_pourvois for jpid in pourvoi_to_jpid.get(p, [])}

        ranked_jp = list(jp_order[np.argsort(-sim_jp[qi])])
        for k in config.KS:
            rows.append({"question_id": qid, "side": "jp",
                          "metric": "pourvoi_resolved", "k": k,
                          "recall": recall_at_k(ranked_jp, jp_gold_ids, k)})
        kstar_j = kstar(ranked_jp, jp_gold_ids, config.KS, config.KSTAR_THRESHOLD)

        kstar_summary.append({
            "question_id":   qid,
            "n_gold_oblig":  len(oblig),
            "kstar_article": kstar_a,
            "n_gold_jp":     len(jp_gold_ids),
            "n_gold_jp_extractible": len(jp_gold_pourvois),
            "kstar_jp":      kstar_j,
        })

    pd.DataFrame(rows).to_csv(config.RECALL_CURVES, index=False)
    config.RECALL_KSTAR.write_text(json.dumps({
        "ks":            list(config.KS),
        "threshold":     config.KSTAR_THRESHOLD,
        "per_question":  kstar_summary,
    }, ensure_ascii=False, indent=2))
    print(f"✓ {config.RECALL_CURVES}")
    print(f"✓ {config.RECALL_KSTAR}")
    print("\nRécap K* par question :")
    for s in kstar_summary:
        print(f"  {s['question_id']:20s}  K*_art={s['kstar_article']:>4}  "
              f"K*_jp={str(s['kstar_jp']):>5}  (gold_oblig={s['n_gold_oblig']}, "
              f"gold_jp={s['n_gold_jp']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 8.6 : Lancer l'éval réelle**

```bash
PYTHONPATH=. python scripts/04_eval_recall.py
```
Attendu (durée ~30 s) :
```
Encodage de 8 questions…
Construction map pourvoi → jp_id…
  ~50000 pourvois CC indexables
✓ data/recall_curves.csv
✓ data/recall_kstar.json
Récap K* par question :
  CNB-PENAL-2025-Q1     K*_art=...  K*_jp=...  (gold_oblig=..., gold_jp=...)
  …
```

- [ ] **Step 8.7 : Commit**

```bash
git add etape1/eval_recall.py tests/test_eval_recall.py scripts/04_eval_recall.py
git commit -m "feat(etape1): eval recall@K + K* + résolution pourvoi-regex (TDD)"
```

---

## Task 9 : Orchestration `run_all.sh` + finalisation

**Files:**
- Create: `run_all.sh`
- Modify: `README.md` (récap résultats)

- [ ] **Step 9.1 : Écrire `run_all.sh`**

```bash
#!/usr/bin/env bash
# Orchestration end-to-end de l'Étape 1.
# Pré-requis : ./scripts/_setup_legi.sh exécuté au moins une fois.
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD"

if [ ! -f data/legi/legi.sqlite ]; then
  echo "→ Build SQLite LEGI (one-shot)"
  ./scripts/_setup_legi.sh
fi

python scripts/02_fetch_articles.py
python scripts/01_token_stats.py
python scripts/03_embed.py --device mps --batch 32
python scripts/04_eval_recall.py

echo
echo "✓ Étape 1 complète. Artefacts dans data/ :"
ls -lh data/*.npy data/*.parquet data/*.csv data/*.json 2>/dev/null
```

```bash
chmod +x run_all.sh
```

- [ ] **Step 9.2 : Compléter le `README.md` avec la table des résultats**

Ajouter à `README.md` une section finale `## Résultats` à remplir manuellement après run avec :
- couverture pair_key→LEGI (global / gold)
- p50/p99/p100 articles, p50/p99/p100 JP-summary
- table des 8 questions × K*_art × K*_jp

- [ ] **Step 9.3 : Smoke final**

```bash
PYTHONPATH=. pytest -v
```
Attendu : tous PASS (config + normalize + resolve + tokenize_stats + linkage + eval_recall).

- [ ] **Step 9.4 : Commit final**

```bash
git add run_all.sh README.md
git commit -m "feat(etape1): orchestration run_all.sh + README résultats"
```

---

## Self-review checklist

- [x] **Spec coverage** — chaque étage du design (Diagnostic / Ingestion+Linkage / Embedding / Eval) couvert par au moins une tâche (Task 5 / Task 3-4-6 / Task 7 / Task 8). Linkage explicite ✓. Diagnostic-first ✓ (Task 5 lit `articles_penal.parquet` produit par Task 4, mais ne dépend pas de l'embedding). Décisions pinées (modèle/scope/K*) ✓.
- [x] **Placeholder scan** — aucun TBD/TODO ; chaque step contient le code complet ; commandes avec sortie attendue.
- [x] **Type consistency** — `pair_key:str`, `article_ids:np.ndarray[object]`, `pairkey_to_graphcol:np.int32`, `KS:list[int]` cohérents entre tasks ; `compute_token_stats / recall_at_k / kstar` mêmes signatures du test à l'impl.

---

## Limitations héritées du design (rappelées ici pour l'exécutant)

- Plafond `gold ⊆ graph_nodes` = 98 % (50/51), `code_penal:222-26-2` inatteignable par construction.
- Couverture `pair_key`→LEGI mesurée à Task 4 — si < 85 %, étendre `normalize.py` avant d'embedder.
- Éval JP biaisée : `jp_attendues.ref = null` → résolution pourvoi-regex CC-only ; 1 question sur 8 a 0 JP gold.
- LEGI = droit actuel : un article abrogé peut ne plus exister → comptabilisé dans `articles_coverage.json`.
