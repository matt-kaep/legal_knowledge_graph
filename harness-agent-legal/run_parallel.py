#!/usr/bin/env python3
"""Runner parallèle du harnais agents-legal.

Distribue les jobs (condition, modalité, question) sur N workers Python, chacun
lançant un process pi indépendant (via run_agents.run_one). Reprise auto : un
job dont le fichier de sortie existe est sauté.

Usage:
  OPENLEGY_TOKEN=... python3 run_parallel.py \
      --questions questions_50.tsv --workers 20 --timeout 300 \
      --conditions web,openlegy --modalities articles,jurisprudence
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
import run_agents

os.environ.setdefault("OPENLEGY_TOKEN", "")


def load_qids_q(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        qid, _, enonce = line.partition("\t")
        rows.append({"qid": qid.strip(), "question": enonce.strip()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=HERE / "questions_50.tsv")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--conditions", default="web,openlegy")
    ap.add_argument("--modalities", default="articles,jurisprudence")
    ap.add_argument("--status-after", type=int, default=20, help="écrit un rapport d'avancement tous les N jobs")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    modals = [m.strip() for m in args.modalities.split(",") if m.strip()]
    questions = load_qids_q(args.questions)
    print(f"questions={len(questions)} conditions={conditions} modalités={modals} workers={args.workers} timeout={args.timeout}")

    jobs = []
    for cond in conditions:
        for q in questions:
            for modal in modals:
                jobs.append({"condition": cond, "qid": q["qid"], "modalite": modal, "question": q["question"]})

    n_done, n_skip, n_err, n_ok, n_timeout = 0, 0, 0, 0, 0
    started_wall = time.monotonic()
    report_path = HERE / f"run_parallel_{int(started_wall)}.jsonl"

    def work(job):
        r = run_agents.run_one(job["condition"], job["qid"], job["modalite"],
                               job["question"], run_agents.PROMTPS[job["modalite"]],
                               timeout=args.timeout)
        return {**job, **r}

    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futures)):
            try:
                res = fut.result()
            except Exception as e:
                res = {"error": f"exception: {e}"}
            n_done += 1
            status = res.get("status")
            if "skipped" in res or status == "skipped":
                status = "skipped"
                n_skip += 1
            elif status == "ok":
                n_ok += 1
            elif status == "timeout":
                n_timeout += 1
                # on garde quand même le résultat (output partiel) dans le log
            elif status == "error":
                n_err += 1
            # journalisation linéaire (append) pour éviter les conflits
            with report_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({k: res.get(k) for k in
                                    ("condition", "qid", "modalite", "status", "turns",
                                     "tool_calls", "refs", "n_refs_csv", "elapsed_seconds",
                                     "timed_out", "error")}, ensure_ascii=False) + "\n")
            if n_done % args.status_after == 0 or n_done == len(jobs):
                wall = time.monotonic() - started_wall
                print(f"[{n_done}/{len(jobs)}] ok={n_ok} timeout={n_timeout} err={n_err} skip={n_skip} "
                      f"wall={wall/60:.1f}min", flush=True)

    wall = time.monotonic() - started_wall
    print("\n=== TERMINÉ ===")
    print(f"jobs={len(jobs)} done={n_done} ok={n_ok} timeout={n_timeout} errors={n_err} skipped={n_skip}")
    print(f"temps total (wall): {wall/60:.1f} min  |  par job moyen: {wall/max(n_done,1):.1f}s")
    print(f"rapport: {report_path}")
    # vérifie la couverture
    coverage = {"web": {"articles": [], "jurisprudence": []},
                "openlegy": {"articles": [], "jurisprudence": []}}
    for c in conditions:
        for m in modals:
            files = sorted((run_agents.OUT / c).glob(f"*__{m}.json"))
            cov = len([f for f in files if f.is_file() and ".trace." not in f.name])
            coverage[c][m] = cov
    print("couverture fichiers:", json.dumps(coverage))


if __name__ == "__main__":
    main()