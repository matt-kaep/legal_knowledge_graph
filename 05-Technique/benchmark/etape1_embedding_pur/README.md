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
