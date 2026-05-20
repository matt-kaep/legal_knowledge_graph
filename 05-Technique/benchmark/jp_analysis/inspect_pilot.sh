#!/usr/bin/env bash
# inspect_pilot.sh — récap du run pilote pour alimenter PILOT.md.
# Usage :  bash inspect_pilot.sh [outputs/step1_pilot]
set -u
OUT="${1:-outputs/step1_pilot}"

if [ ! -d "$OUT" ]; then
  echo "ERREUR : $OUT inexistant" >&2; exit 1
fi

# Active le venv si on n'est pas dedans (pour avoir python3 + json)
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$HOME/.venv-jp-analysis/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.venv-jp-analysis/bin/activate"
fi

python3 - "$OUT" <<'PYEOF'
import json, sys, statistics, collections, pathlib

out = pathlib.Path(sys.argv[1])
shards = sorted(out.glob("*/part-*.jsonl"))
recs = []
for s in shards:
    for line in s.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try: recs.append(json.loads(line))
            except json.JSONDecodeError: pass

print(f"\n==============================================================")
print(f" PILOTE — {len(recs)} records dans {out}")
print(f"==============================================================\n")

# Status × juris
sxj = collections.Counter((r.get("juris","?"), r.get("status","?")) for r in recs)
print("Status × juris :")
print(f"  {'juris':<5}  {'ok':>4} {'failed_terminal':>16} {'oversized':>10} {'no_fulltext':>12}")
for j in ("CC","CA","TJ","?"):
    if not any(jj==j for jj,_ in sxj): continue
    print(f"  {j:<5}  {sxj[(j,'ok')]:>4} {sxj[(j,'failed_terminal')]:>16} "
          f"{sxj[(j,'oversized')]:>10} {sxj[(j,'no_fulltext')]:>12}")

# Themes valid + anomalies
ok = [r for r in recs if r.get("status")=="ok"]
tv = collections.Counter(r.get("themes_valid") for r in ok)
print(f"\nQualité thèmes (sur {len(ok)} ok) :  themes_valid=True : {tv[True]}  "
      f"False : {tv[False]}  None/?: {tv[None]}")
anom = out / "_themes_anomalies.jsonl"
if anom.exists():
    lines = [l for l in anom.read_text().splitlines() if l.strip()]
    print(f"_themes_anomalies.jsonl : {len(lines)} paires rejetées")
    for l in lines[:5]:
        try:    print("  ", json.loads(l))
        except: pass

# Latence p50/p95/p99 depuis _metrics
metr = out / "_metrics.jsonl"
if metr.exists():
    durs = []
    for l in metr.read_text().splitlines():
        if not l.strip(): continue
        try:
            d = json.loads(l).get("duration_ms")
            if d is not None: durs.append(int(d))
        except: pass
    if durs:
        durs.sort()
        def q(p): return durs[min(len(durs)-1, int(len(durs)*p))]
        print(f"\nLatence par record (ms) :  p50={q(.5)}  p95={q(.95)}  p99={q(.99)}  "
              f"min={durs[0]}  max={durs[-1]}  n={len(durs)}")

# Tokens in/out moyens
toks = [(r.get("tokens_in"), r.get("tokens_out")) for r in ok if r.get("tokens_in")]
if toks:
    ins  = [t[0] for t in toks if t[0] is not None]
    outs = [t[1] for t in toks if t[1] is not None]
    if ins:  print(f"Tokens in  : moy={sum(ins)//len(ins)}  max={max(ins)}")
    if outs: print(f"Tokens out : moy={sum(outs)//len(outs)}  max={max(outs)}")

# Un exemple ok lisible (le premier CC ok si possible, sinon le premier ok)
sample = next((r for r in ok if r.get("juris")=="CC"), None) or (ok[0] if ok else None)
if sample:
    print(f"\n=== Exemple record ok (juris={sample.get('juris')}, id={sample.get('id')}) ===")
    for k in ("contexte","dispositif_nature","solution_resume",
              "synthese_pour_avocat","attendu_cle","themes","cited_articles"):
        v = sample.get(k); s = repr(v) if not isinstance(v,str) else v
        if isinstance(s,str) and len(s) > 400: s = s[:400] + " …[tronqué]"
        print(f"\n  {k}:")
        print(f"    {s}")

# Premier failed_terminal si y'en a (pour diagnostiquer)
ft = [r for r in recs if r.get("status")=="failed_terminal"]
if ft:
    print(f"\n=== Exemple failed_terminal ({len(ft)} au total) ===")
    r = ft[0]
    print(f"  id={r.get('id')}  juris={r.get('juris')}  "
          f"error_class={r.get('error_class')}")
    print(f"  error_message={r.get('error_message')}")
PYEOF
