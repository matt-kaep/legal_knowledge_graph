# Étape 1 — Embedding pur

## État actuel (2026-05-26)

### Résumé exécutif

| Composant | Statut |
|---|---|
| Code modules + scripts + tests | ✅ commité sur `etape1-embedding-pur` |
| Tests unitaires | ✅ **26/26 PASS** |
| SQLite LEGI (4.7 GB, 1.7 M articles, snapshot 13/07/2025) | ✅ buildée + indexes |
| `articles_penal.parquet` — 3 183 articles résolus (sur 8 085 attendus) | ✅ |
| Diagnostic couverture (gold **50/50 = 100 %**, global **39 %**) | ✅ |
| `token_stats.json` (articles p100=5214, JP p99=7586) | ✅ |
| `emb_articles.npy` (3 183 × 1 024, fp32, L2) | ✅ **5 min** |
| `emb_jp.npy` (97 060 × 1 024) | 🔄 en cours (~30-50 min) |
| `recall_curves.csv` + `recall_kstar.json` | 🔄 prochaine étape |

### Pivot par rapport au design initial

| Hypothèse design | Réalité | Adaptation |
|---|---|---|
| Embedder JP sur `summary` | Le bundle pénal n'a que `text` (préparation 5 mai) | `text` + filtre `juris=CC` (esprit des 95 % CC summary) |
| `MAX_CTX = 8 192` (BGE-M3 8k) | OOM MPS à seq² × batch trop gros | `BATCH_MAX_LEN = 2 048`, chunk+meanpool au-delà (17 % JP, 5 articles) |
| Couverture LEGI ~95 % | 39 % global (anciens articles pré-1994 renumérotés) | Gold à 100 %, éval valide ; pool candidats 3 183 |
| Modèle e5-base | BGE-M3 (8k contexte) | Cohérent avec design ; impact attendu **× 4** sur signal/JP vs baseline 5 mai |

### Blockers résolus (chaîne `torch < 2.6 → CVE → numpy 2 → transformers 5.x`)

| Symptôme | Cause | Fix |
|---|---|---|
| `legi.py` pip install | hunspell C lib (brew offline ce jour-là) | `pip install --no-deps legi` + `libarchive-c` + `appdirs` |
| FTP DILA stuck `SYN_SENT` | VPN ENST bloque port 21 | `download_legi_via_http` (port 443) + `curl` direct |
| `tar2sqlite` `AssertionError` CID | dump DILA contient mismatches | patch `assert` → `skip` (32 articles + 4 sections perdus) |
| `02_fetch_articles` lent (42 min) | pas d'index sur `articles.num` | `CREATE INDEX idx_articles_num` (~28 s, gain × 1500) |
| Schéma SQL faux | `textes_versions.titre_court` n'existe pas | join sur `articles.cid = textes_versions.id`, filtre `titre` + `etat='VIGUEUR'` |
| `torch.load` CVE-2025-32434 | torch 2.3 trop ancien | upgrade torch 2.12 + torchvision 0.27 + transformers 5.9 |
| OOM MPS attention | seq 8k × batch 32 quadratique | `BATCH_MAX_LEN=2048`, `--batch 8` |

---

## Reprise / commande end-to-end

```bash
cd 05-Technique/benchmark/etape1_embedding_pur
source .venv/bin/activate
PYTHONPATH=. python scripts/03_embed.py --device mps --batch 8   # ~50 min total
PYTHONPATH=. python scripts/04_eval_recall.py                    # ~30 s
```

### Étape par étape

```bash
./scripts/_setup_legi.sh        # une seule fois (~30-60 min DL + ~1h30 build SQLite)
PYTHONPATH=. python scripts/02_fetch_articles.py    # ~1 min
PYTHONPATH=. python scripts/01_token_stats.py       # ~2 min
PYTHONPATH=. python scripts/03_embed.py --device mps --batch 8
PYTHONPATH=. python scripts/04_eval_recall.py
```

## Cluster L40S (repli)

```bash
PYTHONPATH=. python scripts/03_embed.py --device cuda --batch 64
```

## Mémoire & disque

- LEGI dump tar.gz : ~1.1 GB (téléchargé une fois, gardé en `data/legi/fonds/`)
- LEGI SQLite : 4.7 GB (avec indexes)
- HF cache BGE-M3 : ~4 GB (`~/.cache/huggingface/hub/models--BAAI--bge-m3`)
- Embeddings : ~12 MB (articles) + ~400 MB (JP)
