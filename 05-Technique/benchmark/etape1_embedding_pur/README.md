# Étape 1 — Embedding pur

## État actuel (2026-05-20)

- ✅ **Code complet** : 9 tâches du `PLAN.md` implémentées (modules `etape1/` + scripts CLI + `run_all.sh`).
- ✅ **Tests unitaires : 25/25 passent** (`PYTHONPATH=. pytest tests/`).
- 🔄 **Reste à exécuter** : pipeline end-to-end (download LEGI + embedding + éval).
  Tentative du 2026-05-20 bloquée par instabilité du FTP DILA et de brew (réseau dégradé).
  À reprendre quand le réseau est sain.

### Commande de reprise
```bash
cd 05-Technique/benchmark/etape1_embedding_pur
source .venv/bin/activate
./scripts/_setup_legi.sh        # ~30-60 min, télécharge ~3 GB FTP DILA puis build SQLite (~30 GB)
./run_all.sh                    # enchaîne 02→01→03→04 (~25-40 min total)
```

> ⚠ **Note** : `hunspell` (lib C requise par `legi.py` pour le spell-check) n'a pas pu
> être installé via brew sur ce Mac. L'install Python utilise `pip install --no-deps legi`
> + `libarchive-c` + `appdirs` — `legi.tar2sqlite` fonctionne sans hunspell (warning
> bénin). Si brew redevient OK : `brew install hunspell && pip install hunspell` pour
> retrouver les fonctions d'analyse linguistique de legi.py (non utilisées par Étape 1).

---

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
