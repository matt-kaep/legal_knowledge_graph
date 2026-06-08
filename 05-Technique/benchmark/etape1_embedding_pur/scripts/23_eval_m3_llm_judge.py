"""M3 — LLM-as-judge sur les rankings des méthodes champions.

Pour chaque triplet (question q, item i du top-K d'une méthode), Gemma classe
la pertinence en {n2: pertinent, n1: proche, n0: non pertinent}. Agrégation :

    M3(q) = (2·#n2 + #n1) / (2·K_eff)        ∈ [0, 1]
    M3    = moyenne sur les questions

K_eff = nombre d'items réellement jugés dans le ranking de q (≤ K=10).

M3 capte la pertinence sémantique du top-K INDÉPENDAMMENT de la GT — crucial
car la GT est sparse (|GT| moy ≈ 1,23) : on récupère ainsi les vrais positifs
absents de la GT (cf. handoff M3-LLM-judge).

────────────────────────────────────────────────────────────────────────────
Architecture (steers advisor) :
  - worklist DÉDUPLIQUÉE : ensemble unique (qid, item_id, modality) sur toutes
    les méthodes → chaque paire jugée une seule fois (cache inter-méthodes).
  - appels MONO-ITEM concurrents (ThreadPool) → le débit vient du continuous
    batching de vLLM, pas du stuffing de prompt. Pas d'overflow contexte (1 Q +
    1 article tient largement dans 16k), pas de mélange d'indices.
  - cache WRITE-THROUGH (CSV append sous lock) → reprise après crash.
  - response_format json_schema strict + label enum + temperature=0 → juge
    reproductible, piège JSON neutralisé.
  - texte article = HTML → strip avant jugement.

Modes :
  --mock           : labels déterministes (test plomberie local, SANS GPU).
  --pilot N        : ne juger que N questions (incl. cas sanity GT-singleton).
  --methods ...    : sous-ensemble de méthodes champions.

Le run réel tourne sur le NŒUD GPU du cluster (vLLM Gemma localhost), via
run_m3_judge_on_cluster.sh. Ce script ne fait PAS de pip install.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# REPO surchargeable par env LKG_REPO (portabilité Mac dev ↔ nœud GPU cluster).
# config.py utilise des chemins relatifs (ROOT = parents) → reste portable.
REPO = Path(os.environ.get(
    "LKG_REPO",
    "/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph"))
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
from etape1 import config  # noqa: E402

HERE = Path(__file__).resolve().parent
ETAPE1 = HERE.parent  # 05-Technique/benchmark/etape1_embedding_pur
PROMPT_ART = ETAPE1 / "prompts" / "m3_judge_art_v1.txt"
PROMPT_JP = ETAPE1 / "prompts" / "m3_judge_jp_v1.txt"
SCHEMA_PATH = ETAPE1 / "schemas" / "m3_judge_format.json"

OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"
RANKINGS_PARQUET = OUT_DIR / "rankings.parquet"
BENCH_PATH = OUT_DIR / "bench_global.json"
CACHE_CSV = OUT_DIR / "m3_judge_cache.csv"        # write-through (qid,item,modality,label,raison)
EVAL_CSV = OUT_DIR / "eval_m3.csv"                # M3(q) par méthode
SUMMARY_JSON = OUT_DIR / "eval_m3_summary.json"   # agrégats par méthode

K = 10
LABEL_WEIGHT = {"n2": 2, "n1": 1, "n0": 0}

# Méthodes champions à juger (handoff M3). (method, k_in, modality).
# k_in=None → match les lignes rankings dont k_in est NaN (B2-a, B3-a).
CHAMPIONS: list[tuple[str, int | None, str]] = [
    # ── côté ARTICLES
    ("B3-e",         10,   "art"),   # P1 — champion strict (M1=0.711)
    ("PPR-row-a0.95", 10,  "art"),   # P1 — champion étendu (M1ext=0.441)
    ("B2-a",         None, "art"),   # P2 — cosine pur
    ("PPR-row-a0.85", 10,  "art"),   # P2 — variante PPR
    # ── côté JP
    ("B4-e",         20,   "jp"),    # P1 — champion JP (M1=0.416)
    ("B3-a",         None, "jp"),    # P1 — cosine JP direct
    ("B4-d",         50,   "jp"),    # P2 — intersection
    ("B4-f",         10,   "jp"),    # P2 — référence négative (M1=0.154)
    ("PPR-row-a0.95", 10,  "jp"),    # P2 — saturation côté JP
]


# ════════════════════════════════════════════════════════════════════════
# Chargement
# ════════════════════════════════════════════════════════════════════════
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def strip_html(s: str) -> str:
    """Retire les tags HTML (<br/>, <p>…) + dé-échappe les entités."""
    s = _TAG_RE.sub(" ", s or "")
    s = html.unescape(s)
    return _WS_RE.sub(" ", s).strip()


def load_questions() -> dict[str, dict]:
    d = json.loads(BENCH_PATH.read_text())
    out = {}
    for q in d["questions"]:
        out[q["qid"]] = {
            "enonce": q["enonce"],
            "gt": set(q.get("articles_attendus") or []),
        }
    return out


def load_texts() -> tuple[dict[str, str], dict[str, str]]:
    art_df = pd.read_parquet(config.ARTICLES_PARQUET_ALL, columns=["pair_key", "texte"])
    jp_df = pd.read_parquet(config.JP_SUMMARIES_PARQUET, columns=["jp_id", "synthese"])
    art_text = {pk: strip_html(t) for pk, t in zip(art_df["pair_key"], art_df["texte"])}
    jp_text = {j: strip_html(t) for j, t in zip(jp_df["jp_id"], jp_df["synthese"])}
    return art_text, jp_text


def select_rankings(df: pd.DataFrame, methods: set[str] | None) -> pd.DataFrame:
    """Filtre rankings.parquet aux (method, k_in, modality) champions."""
    keep = []
    for method, k_in, modality in CHAMPIONS:
        if methods is not None and method not in methods:
            continue
        m = (df["method"] == method) & (df["modality"] == modality)
        m &= df["k_in"].isna() if k_in is None else (df["k_in"] == k_in)
        sub = df[m].copy()
        if sub.empty:
            print(f"  ⚠ aucune ligne pour ({method}, k_in={k_in}, {modality}) "
                  f"— méthode absente de rankings.parquet ?")
        keep.append(sub)
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[0:0]


# ════════════════════════════════════════════════════════════════════════
# Cache write-through
# ════════════════════════════════════════════════════════════════════════
class JudgeCache:
    """Cache (qid,item_id,modality) -> (label,raison), persistant et thread-safe.

    Append immédiat à chaque label reçu : un run de 10h reprend après crash.
    """
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.store: dict[tuple[str, str, str], tuple[str, str]] = {}
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    self.store[(r["qid"], r["item_id"], r["modality"])] = (
                        r["label"], r.get("raison", ""))
            print(f"  cache HIT : {len(self.store)} jugements déjà présents")
        self._fh = path.open("a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        if path.stat().st_size == 0:
            self._w.writerow(["qid", "item_id", "modality", "label", "raison"])
            self._fh.flush()

    def get(self, key):
        return self.store.get(key)

    def put(self, key, label, raison):
        with self.lock:
            if key in self.store:
                return
            self.store[key] = (label, raison)
            self._w.writerow([key[0], key[1], key[2], label, raison])
            self._fh.flush()

    def close(self):
        self._fh.close()


# ════════════════════════════════════════════════════════════════════════
# Jugement (vLLM Gemma ou mock)
# ════════════════════════════════════════════════════════════════════════
def make_judge(args, schema):
    """Retourne une fonction judge(question, document, modality) -> (label, raison)."""
    if args.mock:
        def judge_mock(question, document, modality):
            h = hashlib.md5((question[:40] + "|" + document[:40]).encode()).hexdigest()
            label = ["n2", "n1", "n0"][int(h, 16) % 3]
            return label, f"[mock] {modality}"
        return judge_mock

    import openai
    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1",
                    api_key="EMPTY", timeout=args.req_timeout, max_retries=0)

    def judge_vllm(question, document, modality):
        template = TEMPLATES[modality]
        prompt = template.replace("{question}", question).replace("{document}", document)
        for attempt in (1, 2):
            try:
                resp = client.chat.completions.create(
                    model=args.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=args.max_tokens_out,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "m3_judge", "schema": schema,
                                        "strict": True},
                    },
                )
                txt = resp.choices[0].message.content
                obj = _parse_label(txt)
                if obj is not None:
                    return obj
            except openai.BadRequestError as e:
                return "n0", f"[vllm_400] {e}"          # contexte → traité non-pertinent
            except (openai.APITimeoutError, openai.APIConnectionError):
                if attempt == 2:
                    return None
        return None

    return judge_vllm


def _parse_label(txt: str):
    """Extrait (label, raison) du JSON, avec fallback regex si bavardage."""
    if not txt:
        return None
    try:
        o = json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            return None
        try:
            o = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    label = str(o.get("label", "")).strip().lower()
    if label not in LABEL_WEIGHT:
        return None
    return label, str(o.get("raison", ""))[:300]


TEMPLATES: dict[str, str] = {}


# ════════════════════════════════════════════════════════════════════════
# Agrégation M3
# ════════════════════════════════════════════════════════════════════════
def aggregate(sel: pd.DataFrame, cache: JudgeCache, questions: dict,
              denom: str = "k_fixed") -> tuple:
    """Construit M3(q) par méthode + distribution (n2,n1,n0) + sanity.

    Dénominateur (cf. discussion gate) :
      - k_fixed   : 2·K, K=10 fixe (déf gelée Week-9). Un ranking de 6 items
                    parfaits plafonne à 0,6 → pénalise les rankings courts.
      - k_ranking : 2·(taille du ranking, ≤10). Un ranking de 6 items parfaits
                    fait 1,0 → ne pénalise pas la non-saturation du top-K.
    Dans les deux cas, un item présent dans le ranking mais non jugé
    (parse-fail / sans texte) compte 0 au numérateur (jamais retiré du
    dénominateur) → l'échec est pénalisé, pas effacé.
    """
    per_q_rows = []
    method_dist: dict[str, dict] = {}
    sanity_gt_labels: list[str] = []  # label des items GT-singleton présents dans un ranking

    grp = sel.groupby(["method", "k_in", "modality"], dropna=False)
    for (method, k_in, modality), sub in grp:
        dist = {"n2": 0, "n1": 0, "n0": 0}
        m3_vals = []
        for qid, qsub in sub.groupby("qid"):
            # sort_values + drop_duplicates : robuste aux qids dupliqués dans la
            # cohorte (bench_global a ~2 qids en double → ranking dupliqué) et à
            # un éventuel item répété dans un ranking. Garde le meilleur rang.
            qsub = qsub.sort_values("rank").drop_duplicates("item_id")
            n_ranked = len(qsub)            # items uniques du ranking (≤ K), jugés ou non
            labels = []
            for item_id in qsub["item_id"]:
                cached = cache.get((qid, item_id, modality))
                # sanity : item GT-singleton (articles uniquement) ?
                q = questions.get(qid)
                if (modality == "art" and q and len(q["gt"]) == 1
                        and item_id in q["gt"] and cached is not None):
                    sanity_gt_labels.append(cached[0])
                if cached is None:
                    continue               # non jugé → 0 au numérateur, reste au dénom
                lab = cached[0]
                labels.append(lab)
                dist[lab] += 1
            if not labels:
                continue
            denom_k = K if denom == "k_fixed" else n_ranked
            score = sum(LABEL_WEIGHT[l] for l in labels) / (2 * denom_k)
            m3_vals.append(score)
            per_q_rows.append({
                "qid": qid, "method": method, "k_in": k_in, "modality": modality,
                "n_ranked": n_ranked, "n_judged": len(labels),
                "n2": labels.count("n2"), "n1": labels.count("n1"),
                "n0": labels.count("n0"), "m3": score,
            })
        kin_disp = "-" if pd.isna(k_in) else str(int(k_in))
        method_dist[f"{method}|{modality}|kin={kin_disp}"] = {
            "n_q": len(m3_vals),
            "m3_mean": float(np.mean(m3_vals)) if m3_vals else float("nan"),
            **dist,
            "n_judged": sum(dist.values()),
        }
    return per_q_rows, method_dist, sanity_gt_labels


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true",
                   help="labels déterministes (test plomberie SANS GPU)")
    p.add_argument("--pilot", type=int, default=None,
                   help="ne juger que N questions (incl. cas sanity GT-singleton)")
    p.add_argument("--methods", nargs="+", default=None,
                   help="sous-ensemble de méthodes (ex: B3-e PPR-row-a0.95)")
    p.add_argument("--model-id", default="cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
                   help="modèle vLLM (défaut: Gemma 4 26B, cf. doctrine_qgen)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=32,
                   help="appels concurrents (continuous batching vLLM)")
    p.add_argument("--req-timeout", type=int, default=300)
    p.add_argument("--max-tokens-out", type=int, default=256)
    p.add_argument("--cache", default=None,
                   help="chemin cache (défaut: hashé sur prompt+modèle)")
    p.add_argument("--denom", choices=["k_fixed", "k_ranking"], default="k_fixed",
                   help="dénominateur M3 : k_fixed=2·K (K=10, déf gelée Week-9) ; "
                        "k_ranking=2·(taille du ranking, ≤10) — n'impacte que les "
                        "méthodes renvoyant <10 items (B4-d, B3-e).")
    args = p.parse_args()

    t0 = time.time()
    schema = json.loads(SCHEMA_PATH.read_text())
    TEMPLATES["art"] = PROMPT_ART.read_text(encoding="utf-8")
    TEMPLATES["jp"] = PROMPT_JP.read_text(encoding="utf-8")

    # Cache clé sur (prompts + modèle) : éditer un prompt ⇒ nouveau cache ⇒
    # re-jugement propre, pas de contamination des labels de l'ancien prompt.
    sig = hashlib.md5(
        (TEMPLATES["art"] + TEMPLATES["jp"] + args.model_id
         + ("mock" if args.mock else "")).encode()).hexdigest()[:8]
    cache_path = Path(args.cache) if args.cache else \
        CACHE_CSV.with_name(f"m3_judge_cache_{sig}.csv")
    print(f"  cache : {cache_path.name}  (signature prompt+modèle={sig})")

    print("══ Chargement ─────────────────────────────────────────────")
    if not RANKINGS_PARQUET.exists():
        print(f"[FAIL] {RANKINGS_PARQUET} absent — lancer scripts 18 et 20 d'abord.",
              file=sys.stderr)
        return 1
    df = pd.read_parquet(RANKINGS_PARQUET)
    methods_filter = set(args.methods) if args.methods else None
    sel = select_rankings(df, methods_filter)
    questions = load_questions()
    art_text, jp_text = load_texts()
    print(f"  rankings sélectionnés : {len(sel)} lignes "
          f"({sel['method'].nunique()} méthodes)")

    # ── Pilote : restreindre aux N questions (priorité cas sanity GT-singleton)
    if args.pilot:
        sanity_qids = [qid for qid in sel["qid"].unique()
                       if len(questions.get(qid, {}).get("gt", [])) == 1]
        other_qids = [qid for qid in sel["qid"].unique() if qid not in set(sanity_qids)]
        chosen = sanity_qids[:args.pilot // 2] + other_qids[:args.pilot - args.pilot // 2]
        sel = sel[sel["qid"].isin(set(chosen))]
        print(f"  PILOTE : {len(set(chosen))} questions "
              f"({min(len(sanity_qids), args.pilot//2)} sanity GT-singleton)")

    # ── Worklist dédupliquée : (qid, item_id, modality) uniques
    sel = sel.copy()
    sel["text"] = sel.apply(
        lambda r: (art_text if r["modality"] == "art" else jp_text).get(r["item_id"]),
        axis=1)
    missing = sel["text"].isna().sum()
    if missing:
        print(f"  ⚠ {missing} items sans texte (seront ignorés au jugement)")
    work = sel.dropna(subset=["text"])[
        ["qid", "item_id", "modality", "text"]
    ].drop_duplicates(subset=["qid", "item_id", "modality"])
    print(f"  paires uniques à juger : {len(work)} "
          f"(vs {len(sel)} couples méthode×rank → dédup)")

    cache = JudgeCache(cache_path)
    todo = [r for r in work.itertuples(index=False)
            if cache.get((r.qid, r.item_id, r.modality)) is None]
    print(f"  à juger maintenant : {len(todo)} (reste {len(work)-len(todo)} en cache)")

    judge = make_judge(args, schema)

    # ── Exécution concurrente
    if todo:
        print(f"\n══ Jugement ({'MOCK' if args.mock else args.model_id}, "
              f"{args.workers} workers) ──")
        done = 0
        fails = 0
        enonce = {qid: questions[qid]["enonce"] for qid in {r.qid for r in todo}
                  if qid in questions}

        def run_one(r):
            q = enonce.get(r.qid)
            if q is None:
                return r, None
            return r, judge(q, r.text, r.modality)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_one, r) for r in todo]
            for fut in as_completed(futs):
                r, res = fut.result()
                done += 1
                if res is None:
                    fails += 1
                else:
                    cache.put((r.qid, r.item_id, r.modality), res[0], res[1])
                if done % 500 == 0:
                    print(f"  {done}/{len(todo)}  (échecs {fails})  "
                          f"(t={time.time()-t0:.0f}s)")
        print(f"  fini : {done} jugés, {fails} échecs (t={time.time()-t0:.0f}s)")

    # ── Agrégation
    print("\n══ Agrégation M3 ──────────────────────────────────────────")
    per_q, method_dist, sanity = aggregate(sel, cache, questions, denom=args.denom)
    cache.close()
    print(f"  dénominateur M3 : {args.denom} "
          f"({'2·K=20' if args.denom == 'k_fixed' else '2·taille_ranking'})")

    # En mock, suffixer les sorties pour ne PAS polluer les vrais résultats.
    suffix = "_mock" if args.mock else ("_pilot" if args.pilot else "")
    eval_csv = EVAL_CSV.with_name(f"eval_m3{suffix}.csv")
    summary_json = SUMMARY_JSON.with_name(f"eval_m3_summary{suffix}.json")

    pd.DataFrame(per_q).to_csv(eval_csv, index=False)
    print(f"✓ {eval_csv}  ({len(per_q)} lignes qid×méthode)")

    print(f"\n  {'méthode|modal|kin':<26s} {'n_q':>4s} {'M3':>6s} "
          f"{'n2':>6s} {'n1':>6s} {'n0':>6s}")
    print("  " + "─" * 60)
    for key, d in sorted(method_dist.items()):
        print(f"  {key:<26s} {d['n_q']:>4d} {d['m3_mean']:>6.3f} "
              f"{d['n2']:>6d} {d['n1']:>6d} {d['n0']:>6d}")

    # Sanity : les items GT-singleton devraient être massivement n2
    if sanity:
        c2 = sanity.count("n2")
        print(f"\n  ── SANITY (items GT-singleton présents dans un ranking) ──")
        print(f"  {len(sanity)} items GT jugés : "
              f"n2={c2} ({100*c2/len(sanity):.0f}%)  "
              f"n1={sanity.count('n1')}  n0={sanity.count('n0')}")
        print(f"  (prompt bien calibré ⇒ la grande majorité doit être n2)")

    summary_json.write_text(json.dumps({
        "denom": args.denom,
        "methods": method_dist,
        "sanity_gt_singleton": {
            "n": len(sanity), "n2": sanity.count("n2"),
            "n1": sanity.count("n1"), "n0": sanity.count("n0"),
        },
        "model": "mock" if args.mock else args.model_id,
    }, ensure_ascii=False, indent=2))
    print(f"\n✓ {summary_json}")
    print(f"  t total : {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
