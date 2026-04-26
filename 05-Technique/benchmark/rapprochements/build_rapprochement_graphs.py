"""Construction des 3 graphes JP→JP depuis les rapprochements Cour de cassation.

Les trois périmètres — à exécuter dans un même run car la logique est identique :

  (ii) RESSERRÉ : nœuds = 1 532 arrêts-sources du benchmark + leurs rapprochements directs
                  ≈ 5-8 k nœuds, 5 832 arêtes (une arête par rapprochement parsable)

  (i)  LARGE   : nœuds = tous les arrêts qui apparaissent comme source OU cible
                 d'au moins un rapprochement parsable (sur tout le corpus CC)
                 ≈ 70-80 k nœuds, 18 901 arêtes (toutes celles parsables)

  (iii) TOUT CC : nœuds = les 553 075 arrêts CC (même sans rapprochement)
                  arêtes = toutes celles parsables
                  → permet d'analyser la distribution structurelle globale

Pipeline (2 passes streaming) :
  Pass 1 — scan du corpus : pour chaque arrêt, on collecte (id, ecli, chamber,
           date, numbers_norm, rapprochements_parsables → pourvoi_norm).
  Pass 2 — construction des 3 graphes à partir de l'index + export.

Livrables par graphe :
  - data/graphs/rapp-{periscope}.pkl       (NetworkX.DiGraph picklé)
  - data/graphs/rapp-{periscope}.graphml   (ouvrable dans Gephi)
  - data/graphs/metrics-{periscope}.md     (stats détaillées)
  - data/graphs/summary.md                 (comparatif des 3 graphes)

Note : les arêtes sont **dirigées** (source X déclare Y comme rapprochement) mais
       la sémantique "ligne jurisprudentielle" est symétrique — à l'analyse on
       considérera souvent le graphe non dirigé (`G.to_undirected()`).
"""

from __future__ import annotations

import json
import pickle
import re
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

# ── Ré-utilise la logique du script benchmark ──────────────────────────────
from build_rapprochement_benchmark import (
    POURVOI_RE,
    HREF_RE,
    canon_chamber,
    normalize_pourvoi,
    parse_rapprochement,
)

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "database-judilibre-enrichie" / "Cour de cassation"
BENCHMARK_PATH = ROOT / "data" / "rapprochements" / "benchmark-rapp-v1.json"
OUTPUT_DIR = ROOT / "data" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Pass 1 : index global streaming ────────────────────────────────────────

def scan_corpus(input_path: Path) -> tuple[dict, list[tuple[str, str]]]:
    """Renvoie :
       - nodes_meta : {pourvoi_norm -> {id, ecli, chamber, chamber_canon, date, solution}}
       - edges      : liste de (pourvoi_src_norm, pourvoi_tgt_norm) parsables.
    """
    nodes_meta: dict[str, dict] = {}
    edges: list[tuple[str, str]] = []
    stats = {"total": 0, "edges_raw": 0, "edges_parsable": 0, "nodes_with_rapp": 0}
    t0 = time.time()

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["total"] += 1

            # Clé(s) pourvoi source
            numbers = rec.get("numbers") or []
            src_keys = [normalize_pourvoi(n) for n in numbers if n]
            src_keys = [k for k in src_keys if k]
            if not src_keys:
                continue
            src_main = src_keys[0]

            # Enregistrer le nœud (garde premier vu si collision)
            if src_main not in nodes_meta:
                nodes_meta[src_main] = {
                    "id": rec.get("id"),
                    "ecli": rec.get("ecli"),
                    "chamber_raw": rec.get("chamber"),
                    "chamber": canon_chamber(rec.get("chamber")),
                    "date": rec.get("decision_date"),
                    "solution": rec.get("solution"),
                    "publication": rec.get("publication"),
                }

            # Arêtes
            rapps = rec.get("rapprochements") or []
            if rapps:
                stats["nodes_with_rapp"] += 1
            for r in rapps:
                stats["edges_raw"] += 1
                p = parse_rapprochement(r.get("title"))
                if p is None:
                    continue
                tgt = p["pourvoi_norm"]
                if not tgt:
                    continue
                stats["edges_parsable"] += 1
                edges.append((src_main, tgt))
                # Ajoute un node "shadow" minimal pour la cible si on ne la croise
                # pas comme source ailleurs (sera remplacé quand on la verra)
                if tgt not in nodes_meta:
                    nodes_meta[tgt] = {
                        "id": p.get("href_id"),
                        "ecli": None,
                        "chamber_raw": p.get("chamber_hint"),
                        "chamber": canon_chamber(p.get("chamber_hint")),
                        "date": None,
                        "solution": None,
                        "publication": None,
                        "shadow": True,
                    }
            if stats["total"] % 100_000 == 0:
                print(f"  scan — {stats['total']:,} ({time.time()-t0:.1f}s)")

    print(f"\n[scan] {stats['total']:,} records, {stats['nodes_with_rapp']:,} avec rapp.")
    print(f"[scan] arêtes brutes {stats['edges_raw']:,}, parsables {stats['edges_parsable']:,}.")
    print(f"[scan] pourvois indexés (nodes_meta) : {len(nodes_meta):,}")
    return nodes_meta, edges, stats


# ── Construction des 3 graphes ─────────────────────────────────────────────

def build_graph(periscope: str,
                nodes_meta: dict,
                edges: list[tuple[str, str]],
                benchmark_sources: set[str]) -> nx.DiGraph:
    """Construit un DiGraph selon le périmètre :
       - 'resserre' : nœuds = sources benchmark + cibles directes
       - 'large'    : nœuds = tous ceux qui apparaissent dans ≥1 arête
       - 'tout_cc'  : nœuds = toutes les clés de nodes_meta (≈ tout le CC)
    """
    G = nx.DiGraph()
    if periscope == "resserre":
        tgt_set = {t for s, t in edges if s in benchmark_sources}
        keep = benchmark_sources | tgt_set
        eligible_edges = [(s, t) for s, t in edges if s in benchmark_sources]
    elif periscope == "large":
        keep = {s for s, _ in edges} | {t for _, t in edges}
        eligible_edges = edges
    elif periscope == "tout_cc":
        keep = set(nodes_meta.keys())
        eligible_edges = edges
    else:
        raise ValueError(periscope)

    for k in keep:
        meta = nodes_meta.get(k, {})
        # NetworkX n'accepte pas None dans GraphML, on remplace par ""
        attrs = {kk: ("" if vv is None else vv) for kk, vv in meta.items()
                 if not isinstance(vv, (list, dict)) and kk != "shadow"}
        attrs["is_shadow"] = bool(meta.get("shadow"))
        attrs["is_benchmark_src"] = k in benchmark_sources
        G.add_node(k, **attrs)

    for s, t in eligible_edges:
        if s in keep and t in keep:
            if G.has_edge(s, t):
                G[s][t]["weight"] += 1
            else:
                G.add_edge(s, t, weight=1)
    return G


# ── Métriques ──────────────────────────────────────────────────────────────

def compute_metrics(G: nx.DiGraph, name: str, heavy: bool = True) -> dict:
    t0 = time.time()
    Gu = G.to_undirected()
    n, m = G.number_of_nodes(), G.number_of_edges()
    metrics = {
        "name": name,
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G),
        "n_self_loops": nx.number_of_selfloops(G),
    }
    # Composantes
    ccs = list(nx.connected_components(Gu))
    metrics["n_components"] = len(ccs)
    if ccs:
        sizes = sorted((len(c) for c in ccs), reverse=True)
        metrics["largest_cc"] = sizes[0]
        metrics["top5_cc_sizes"] = sizes[:5]

    # Degrés
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    und_deg = dict(Gu.degree())
    metrics["max_in_degree"] = max(in_deg.values()) if in_deg else 0
    metrics["max_out_degree"] = max(out_deg.values()) if out_deg else 0
    metrics["mean_degree"] = sum(und_deg.values()) / max(1, n)
    metrics["isolated_nodes"] = sum(1 for d in und_deg.values() if d == 0)

    # Top centralités (in-degree proxy)
    def fmt_node(k, deg):
        d = G.nodes[k]
        return {
            "pourvoi": k,
            "chamber": d.get("chamber", ""),
            "date": d.get("date", ""),
            "id": d.get("id", ""),
            "degree": deg,
        }
    top_in = sorted(in_deg.items(), key=lambda x: -x[1])[:20]
    metrics["top20_in_degree"] = [fmt_node(k, d) for k, d in top_in if d > 0]

    # PageRank sur graphe non dirigé (proxy meilleur pour lignée jurispr.)
    if heavy and m > 0 and n < 300_000:
        try:
            pr = nx.pagerank(Gu, max_iter=100)
            top_pr = sorted(pr.items(), key=lambda x: -x[1])[:20]
            metrics["top20_pagerank"] = [
                {**fmt_node(k, und_deg.get(k, 0)), "pagerank": round(v, 6)}
                for k, v in top_pr
            ]
        except Exception as e:
            metrics["pagerank_error"] = str(e)

    # Distribution de degrés (binned)
    cnt = Counter(und_deg.values())
    metrics["degree_distribution"] = {
        "0": cnt.get(0, 0),
        "1": cnt.get(1, 0),
        "2-3": sum(cnt.get(i, 0) for i in [2, 3]),
        "4-9": sum(cnt.get(i, 0) for i in range(4, 10)),
        "10-49": sum(cnt.get(i, 0) for i in range(10, 50)),
        "50+": sum(v for k, v in cnt.items() if k >= 50),
    }

    # Répartition par chambre (sur nœuds non-shadow seulement)
    by_chamber = Counter(
        G.nodes[k].get("chamber", "?") for k in G.nodes
        if not G.nodes[k].get("is_shadow")
    )
    metrics["by_chamber_top10"] = dict(by_chamber.most_common(10))

    metrics["compute_seconds"] = round(time.time() - t0, 1)
    return metrics


# ── Export ─────────────────────────────────────────────────────────────────

def export_graph(G: nx.DiGraph, name: str, include_graphml: bool = True) -> dict:
    sizes = {}

    # Pickle (rapide à reloader)
    pkl = OUTPUT_DIR / f"rapp-{name}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(G, f)
    sizes["pickle_mo"] = round(pkl.stat().st_size / 1e6, 2)

    # GraphML (ouvrable Gephi) — certains types doivent être nettoyés
    if include_graphml:
        # Copier en nettoyant les attributs non exportables
        Gx = nx.DiGraph()
        for n, attrs in G.nodes(data=True):
            Gx.add_node(n, **{
                k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in attrs.items()
            })
        for u, v, attrs in G.edges(data=True):
            Gx.add_edge(u, v, **attrs)
        gml = OUTPUT_DIR / f"rapp-{name}.graphml"
        nx.write_graphml(Gx, gml)
        sizes["graphml_mo"] = round(gml.stat().st_size / 1e6, 2)
    return sizes


def write_metrics_md(metrics: dict, sizes: dict) -> Path:
    name = metrics["name"]
    lines = [
        f"# Graphe `{name}` — métriques",
        "",
        "## Synthèse",
        "",
        f"- Nœuds : **{metrics['n_nodes']:,}**",
        f"- Arêtes : **{metrics['n_edges']:,}**",
        f"- Densité : {metrics['density']:.2e}",
        f"- Composantes connexes : {metrics.get('n_components', 0):,} "
        f"(plus grosse : {metrics.get('largest_cc', 0):,})",
        f"- Nœuds isolés : {metrics['isolated_nodes']:,}",
        f"- Degré moyen : {metrics['mean_degree']:.2f}",
        f"- Degré entrant max : {metrics['max_in_degree']} ; sortant max : {metrics['max_out_degree']}",
        f"- Self-loops : {metrics['n_self_loops']}",
        f"- Temps de calcul : {metrics['compute_seconds']} s",
        "",
        "## Distribution de degré (non dirigé)",
        "",
        "| Degré | N nœuds |",
        "|---|---:|",
    ]
    for bucket, n in metrics["degree_distribution"].items():
        lines.append(f"| {bucket} | {n:,} |")

    lines += ["", "## Répartition par chambre (top 10)", ""]
    for ch, n in metrics["by_chamber_top10"].items():
        lines.append(f"- {ch} : {n:,}")

    lines += ["", "## Top 20 — degré entrant (arrêts les plus cités comme rapprochement)", "",
              "| pourvoi | chambre | date | degré_in | id |",
              "|---|---|---|---:|---|"]
    for n in metrics.get("top20_in_degree", []):
        lines.append(f"| {n['pourvoi']} | {n['chamber']} | {n['date']} | {n['degree']} | {n['id']} |")

    if metrics.get("top20_pagerank"):
        lines += ["", "## Top 20 — PageRank (non dirigé) — arrêts-pivots", "",
                  "| pourvoi | chambre | date | degré | pagerank |",
                  "|---|---|---|---:|---:|"]
        for n in metrics["top20_pagerank"]:
            lines.append(f"| {n['pourvoi']} | {n['chamber']} | {n['date']} | {n['degree']} | {n['pagerank']} |")

    lines += ["", "## Exports", "",
              f"- `rapp-{name}.pkl` : {sizes.get('pickle_mo','?')} Mo",
              f"- `rapp-{name}.graphml` : {sizes.get('graphml_mo','?')} Mo"]

    out = OUTPUT_DIR / f"metrics-{name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_summary(all_metrics: list[dict], scan_stats: dict) -> Path:
    lines = [
        "# Graphes rapprochements JP — synthèse comparative",
        "",
        f"Source : `database-judilibre-enrichie/Cour de cassation` — {scan_stats['total']:,} arrêts.",
        f"Arêtes parsables : {scan_stats['edges_parsable']:,} / {scan_stats['edges_raw']:,} brutes "
        f"({scan_stats['edges_parsable']/max(1,scan_stats['edges_raw'])*100:.1f} %).",
        "",
        "## Comparatif des 3 périmètres",
        "",
        "| Périmètre | Nœuds | Arêtes | Densité | Composantes | Isolés | Max degree in | Compute |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['name']} | {m['n_nodes']:,} | {m['n_edges']:,} | {m['density']:.2e} "
            f"| {m['n_components']:,} | {m['isolated_nodes']:,} | {m['max_in_degree']} "
            f"| {m['compute_seconds']} s |"
        )
    lines += ["",
              "## Lecture",
              "",
              "- **resserre** : vue centrée benchmark — les 1 532 arrêts-sources + leurs rapprochements directs.",
              "- **large** : tous les arrêts ayant au moins un lien de rapprochement (source ou cible).",
              "- **tout_cc** : le corpus entier (553 k arrêts) — la plupart sont isolés, mais ça donne la densité réelle.",
              "",
              "## Fichiers générés",
              "",
              "Pour chaque périmètre :",
              "- `rapp-<name>.pkl` (NetworkX picklé, rapide à reloader)",
              "- `rapp-<name>.graphml` (ouvrable dans Gephi)",
              "- `metrics-<name>.md` (métriques détaillées + top 20)"]
    out = OUTPUT_DIR / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Input  : {INPUT_PATH}")
    print(f"Output : {OUTPUT_DIR}")

    # Charger la liste des 1532 arrêts-sources du benchmark (clé pourvoi_norm)
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench = json.load(f)
    benchmark_sources = {
        normalize_pourvoi(q["decision"]["pourvoi"])
        for q in bench["questions"]
        if q["decision"]["pourvoi"]
    }
    print(f"Sources benchmark : {len(benchmark_sources):,}")

    print("\n── Pass 1 : scan du corpus ──────────────────────────────────────")
    nodes_meta, edges, scan_stats = scan_corpus(INPUT_PATH)

    all_metrics = []
    periscopes = [
        ("resserre", True),   # heavy metrics OK
        ("large",    True),
        ("tout_cc",  False),  # skip PageRank (553k nodes → lent)
    ]
    for periscope, heavy in periscopes:
        print(f"\n── Graphe '{periscope}' ────────────────────────────────")
        G = build_graph(periscope, nodes_meta, edges, benchmark_sources)
        print(f"  |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")
        metrics = compute_metrics(G, periscope, heavy=heavy)
        sizes = export_graph(G, periscope, include_graphml=(periscope != "tout_cc"))
        md = write_metrics_md(metrics, sizes)
        all_metrics.append(metrics)
        print(f"  ✅ {md.name} ; pickle {sizes.get('pickle_mo','?')} Mo "
              f"{'+ graphml ' + str(sizes.get('graphml_mo','?')) + ' Mo' if 'graphml_mo' in sizes else ''}")

    summary = write_summary(all_metrics, scan_stats)
    print(f"\n✅ Synthèse : {summary}")

    print("\n=== Récap rapide ===")
    for m in all_metrics:
        print(f"  {m['name']:10s} | nodes={m['n_nodes']:>8,}  edges={m['n_edges']:>6,}  "
              f"components={m['n_components']:>6,}  isolated={m['isolated_nodes']:>7,}")


if __name__ == "__main__":
    main()
