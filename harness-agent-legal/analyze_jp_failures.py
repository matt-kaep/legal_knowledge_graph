#!/usr/bin/env python3
"""Analyse des traces jurisprudence pour comprendre les échecs.

Pour chaque condition (web, openlegy), analyse N traces JP et compte :
  - web  : erreurs web_search (rate limit, 403, network), fetch échoués,
           sites/domaines, nb de résultats par recherche, nb de pourvois extraits
           par la sortie, nb de refs vides.
  - openlegy : nb d'appels d'outils (rechercher_jurisprudence_judiciaire,
           get_decision_judiciaire, etc.), erreurs MCP, JURITEXT/pourvois dans
           les résultats, nb de refs vides.

Usage:
  python3 analyze_jp_failures.py [--max-traces N] [--condition web|openlegy|all]
"""
from __future__ import annotations
import argparse
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QIDS = {l.split("\t")[0] for l in (HERE / "questions_754.tsv").read_text().splitlines() if l.strip()}


def iter_traces(condition, modal, limit):
    d = HERE / "output" / condition
    got = 0
    for f in sorted(d.glob(f"*__{modal}.trace.json")):
        yield f
        got += 1
        if limit and got >= limit:
            break


def parse_web(trace_path):
    st = {
        "search_calls": 0, "search_ok": 0, "search_err": 0,
        "fetch_calls": 0, "fetch_ok": 0, "fetch_err": 0,
        "domains": collections.Counter(), "err_types": collections.Counter(),
        "n_results": 0,
    }
    try:
        lines = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(lines, list):
        return None
    for e in lines:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "tool_execution_end":
            continue
        t = e.get("toolName")
        res = e.get("result")
        s = json.dumps(res) if res is not None else ""
        err = e.get("isError")
        if t == "web_search":
            st["search_calls"] += 1
            if err or "Error" in s[:80]:
                st["search_err"] += 1
                for k in ("rate limit", "429", "unavailable", "network", "403", "timeout"):
                    if k.lower() in s.lower():
                        st["err_types"][f"search:{k}"] += 1
                        break
                else:
                    st["err_types"]["search:other"] += 1
            else:
                st["search_ok"] += 1
                st["n_results"] += len(re.findall(r'"title"', s))
            for m in re.findall(r"https?://([a-z0-9\-\.]+)", s, re.I):
                st["domains"][m.lower()] += 1
        elif t == "fetch_content":
            st["fetch_calls"] += 1
            if err or "Error" in s[:60] or "failed" in s[:60].lower() or "unable" in s[:60].lower():
                st["fetch_err"] += 1
            else:
                st["fetch_ok"] += 1
    return st


def parse_openlegy(trace_path):
    st = {
        "jp_search_calls": 0, "jp_get_calls": 0, "code_calls": 0,
        "err_types": collections.Counter(), "juritext_seen": 0,
        "pourvoi_in_response": 0, "other_tools": collections.Counter(),
    }
    try:
        lines = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(lines, list):
        return None
    for e in lines:
        if not isinstance(e, dict):
            continue
        if e.get("type") != "tool_execution_end":
            continue
        t = e.get("toolName")
        res = e.get("result")
        s = json.dumps(res) if res is not None else ""
        err = e.get("isError")
        if not t:
            continue
        if "jurisprudence" in t or "decision" in t or "rechercher_" in t:
            st["jp_search_calls"] += 1
            if "JURITEXT" in s:
                st["juritext_seen"] += 1
            if re.search(r"n[°o]?\s*\d{1,2}-\d{2}\.\d{3,4}", s):
                st["pourvoi_in_response"] += 1
            if err:
                st["err_types"][f"{t}:error"] += 1
            if "rechercher_code" in t:
                st["code_calls"] += 1
        else:
            st["other_tools"][t] += 1
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-traces", type=int, default=200)
    ap.add_argument("--condition", default="all")
    args = ap.parse_args()

    conds = ["web", "openlegy"] if args.condition == "all" else [args.condition]

    for cond in conds:
        parser = parse_web if cond == "web" else parse_openlegy
        agg = collections.Counter()
        domains = collections.Counter()
        errs = collections.Counter()
        n_parsed = 0
        for path in iter_traces(cond, "jurisprudence", args.max_traces):
            st = parser(path)
            if not st:
                continue
            n_parsed += 1
            for k, v in st.items():
                if isinstance(v, int):
                    agg[k] += v
            if "domains" in st:
                domains.update(st["domains"])
            if "err_types" in st:
                errs.update(st["err_types"])
        print(f"\n=== {cond.upper()} — jurisprudence : {n_parsed} traces analysées ===")
        print("agrégats outils:", dict(agg))
        if errs:
            print("erreurs outils:", dict(errs.most_common(12)))
        if domains:
            print("top domaines:", dict(domains.most_common(10)))


if __name__ == "__main__":
    main()