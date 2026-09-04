"""Phase B — Graphe bipartite Jurisprudence × Articles (Cour de cassation).

Étend les graphes de Phase A (rapprochements JP-JP) en ajoutant les articles de
loi comme seconde classe de nœuds, avec les arêtes `cite` (Decision → Article)
issues du champ `code_article_pairs` déjà normalisé par
`enrichissement_base_complete.ipynb` (cf. [[Format-Fondement-Juridique]]).

Schéma des nœuds :
  type='decision' → attrs : {pourvoi_norm (id), id_judilibre, ecli, chamber,
                             chamber_raw, date, solution, publication,
                             is_benchmark_src, is_shadow}
  type='article'  → attrs : {pair_key (id), code_slug, article_num}

Schéma des arêtes :
  rel='rapproche'  (Decision → Decision) : héritée de Phase A
  rel='cite'       (Decision → Article)  : extraite de code_article_pairs
  poids = nombre de fois où la paire apparaît (dédup sur la même décision)

Trois périmètres, comme Phase A :
  - resserre : 1 532 sources benchmark + cibles directes + articles de ces JP
  - large    : arrêts ayant ≥1 rapprochement source/cible + leurs articles
  - tout_cc  : toutes les 553 k décisions + tous les articles cités

Livrables (data/graphs_bipartite/) : .pkl, .graphml (sauf tout_cc), metrics-*.md,
summary.md.
"""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from build_rapprochement_benchmark import (
    canon_chamber,
    normalize_pourvoi,
    parse_rapprochement,
)

ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "database-judilibre-enrichie" / "Cour de cassation"
BENCHMARK_PATH = ROOT / "data" / "rapprochements" / "benchmark-rapp-v1.json"
OUTPUT_DIR = ROOT / "data" / "graphs_bipartite"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────

def split_pair_key(pair_key: str) -> tuple[str, str] | None:
    """`code_civil:1240` -> ('code_civil', '1240'). Format canonique issu de
    `enrichissement_base_complete.ipynb`."""
    if not pair_key or ":" not in pair_key:
        return None
    slug, num = pair_key.split(":", 1)
    slug = slug.strip()
    num = num.strip()
    if not slug or not num:
        return None
    return slug, num


# ── Pass 1 : scan du corpus ────────────────────────────────────────────────

def scan_corpus(input_path: Path) -> dict:
    """Retourne un dict avec :
       - nodes_jp : {pourvoi_norm -> meta}
       - edges_rapp : liste de (src, tgt) pourvois
       - edges_cite : liste de (pourvoi_norm, pair_key)
       - articles   : {pair_key -> {code_slug, article_num, n_citations}}
       - stats
    """
    nodes_jp: dict[str, dict] = {}
    edges_rapp: list[tuple[str, str]] = []
    edges_cite: list[tuple[str, str]] = []  # (pourvoi_norm, pair_key)
    articles: dict[str, dict] = {}
    stats = {
        "total": 0, "with_pairs": 0, "with_rapp": 0,
        "pairs_seen": 0, "pair_keys_unique": 0,
        "rapp_brut": 0, "rapp_parsable": 0,
        "pair_dropped_malformed": 0,
    }
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
            numbers = rec.get("numbers") or []
            src_keys = [normalize_pourvoi(n) for n in numbers if n]
            src_keys = [k for k in src_keys if k]
            if not src_keys:
                continue
            src = src_keys[0]
            if src not in nodes_jp:
                nodes_jp[src] = {
                    "type": "decision",
                    "id_judilibre": rec.get("id") or "",
                    "ecli": rec.get("ecli") or "",
                    "chamber_raw": rec.get("chamber") or "",
                    "chamber": canon_chamber(rec.get("chamber")),
                    "date": rec.get("decision_date") or "",
                    "solution": rec.get("solution") or "",
                    "publication": ";".join(rec.get("publication") or []) or "",
                    "shadow": False,
                }

            # code_article_pairs
            pairs = rec.get("code_article_pairs") or []
            if pairs:
                stats["with_pairs"] += 1
                seen_this_rec = set()  # dédup à l'intérieur du même arrêt
                for pk in pairs:
                    stats["pairs_seen"] += 1
                    sp = split_pair_key(pk)
                    if sp is None:
                        stats["pair_dropped_malformed"] += 1
                        continue
                    code_slug, article_num = sp
                    if pk not in articles:
                        articles[pk] = {
                            "type": "article",
                            "pair_key": pk,
                            "code_slug": code_slug,
                            "article_num": article_num,
                            "n_citations": 0,
                        }
                    if pk in seen_this_rec:
                        continue
                    seen_this_rec.add(pk)
                    articles[pk]["n_citations"] += 1
                    edges_cite.append((src, pk))

            # Rapprochements
            rapps = rec.get("rapprochements") or []
            if rapps:
                stats["with_rapp"] += 1
            for r in rapps:
                stats["rapp_brut"] += 1
                p = parse_rapprochement(r.get("title"))
                if p is None:
                    continue
                stats["rapp_parsable"] += 1
                tgt = p["pourvoi_norm"]
                if not tgt:
                    continue
                edges_rapp.append((src, tgt))
                if tgt not in nodes_jp:
                    nodes_jp[tgt] = {
                        "type": "decision",
                        "id_judilibre": p.get("href_id") or "",
                        "ecli": "",
                        "chamber_raw": p.get("chamber_hint") or "",
                        "chamber": canon_chamber(p.get("chamber_hint")),
                        "date": "",
                        "solution": "",
                        "publication": "",
                        "shadow": True,
                    }

            if stats["total"] % 100_000 == 0:
                print(f"  scan — {stats['total']:,} records, "
                      f"{len(articles):,} articles, {len(edges_cite):,} edges cite "
                      f"({time.time()-t0:.1f}s)")

    stats["pair_keys_unique"] = len(articles)
    print(f"\n[scan] {stats['total']:,} records.")
    print(f"[scan] JP indexés : {len(nodes_jp):,} (dont shadows: "
          f"{sum(1 for v in nodes_jp.values() if v.get('shadow')):,})")
    print(f"[scan] Articles uniques : {len(articles):,}")
    print(f"[scan] Arêtes cite : {len(edges_cite):,}")
    print(f"[scan] Arêtes rapproche : {len(edges_rapp):,}")
    return {
        "nodes_jp": nodes_jp,
        "edges_rapp": edges_rapp,
        "edges_cite": edges_cite,
        "articles": articles,
        "stats": stats,
    }


# ── Construction ───────────────────────────────────────────────────────────

def build_graph(periscope: str, data: dict, benchmark_sources: set[str]) -> nx.DiGraph:
    G = nx.DiGraph()
    nodes_jp = data["nodes_jp"]
    edges_rapp = data["edges_rapp"]
    edges_cite = data["edges_cite"]
    articles = data["articles"]

    # Ensemble des JP à conserver
    if periscope == "resserre":
        jp_keep = set(benchmark_sources)
        # + cibles de rapprochements émis par benchmark_sources
        jp_keep |= {t for s, t in edges_rapp if s in benchmark_sources}
    elif periscope == "large":
        jp_keep = {s for s, _ in edges_rapp} | {t for _, t in edges_rapp}
    elif periscope == "tout_cc":
        jp_keep = set(nodes_jp.keys())
    else:
        raise ValueError(periscope)

    # Arêtes cite conservées (dont la source est dans jp_keep)
    cite_in = [(s, pk) for s, pk in edges_cite if s in jp_keep]
    articles_keep = {pk for _, pk in cite_in}

    # Ajouter JP
    for k in jp_keep:
        meta = dict(nodes_jp.get(k, {"type": "decision"}))
        meta["pourvoi_norm"] = k
        meta["is_benchmark_src"] = k in benchmark_sources
        G.add_node(k, **meta)

    # Ajouter Articles
    for pk in articles_keep:
        a = articles[pk]
        G.add_node(pk, **{
            "type": "article",
            "pair_key": pk,
            "code_slug": a["code_slug"],
            "article_num": a["article_num"],
            "n_citations_corpus": a["n_citations"],
        })

    # Arêtes rapproche (conservées seulement si les 2 extrémités sont dans jp_keep)
    for s, t in edges_rapp:
        if s in jp_keep and t in jp_keep:
            if G.has_edge(s, t):
                G[s][t]["weight"] += 1
            else:
                G.add_edge(s, t, rel="rapproche", weight=1)

    # Arêtes cite
    for s, pk in cite_in:
        if G.has_edge(s, pk):
            G[s][pk]["weight"] += 1
        else:
            G.add_edge(s, pk, rel="cite", weight=1)

    return G


# ── Métriques ──────────────────────────────────────────────────────────────

def compute_metrics(G: nx.DiGraph, name: str) -> dict:
    t0 = time.time()
    n = G.number_of_nodes()
    m = G.number_of_edges()

    n_jp = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "decision")
    n_art = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "article")
    m_rapp = sum(1 for _, _, d in G.edges(data=True) if d.get("rel") == "rapproche")
    m_cite = sum(1 for _, _, d in G.edges(data=True) if d.get("rel") == "cite")

    # Connexité sur graphe non dirigé (pour mesurer la connectivité effective)
    Gu = G.to_undirected()
    ccs = list(nx.connected_components(Gu))
    largest = max((len(c) for c in ccs), default=0)

    # Top articles les plus cités dans ce périmètre (degré entrant des nœuds type=article)
    art_in_deg = [(n, G.in_degree(n)) for n, d in G.nodes(data=True)
                  if d.get("type") == "article"]
    art_in_deg.sort(key=lambda x: -x[1])
    top_articles = []
    for pk, deg in art_in_deg[:30]:
        d = G.nodes[pk]
        top_articles.append({
            "pair_key": pk,
            "code_slug": d["code_slug"],
            "article_num": d["article_num"],
            "in_degree": deg,
        })

    # Top décisions les plus couplées (degré sortant = nb articles cités)
    jp_out_deg = [(n, G.out_degree(n)) for n, d in G.nodes(data=True)
                  if d.get("type") == "decision"]
    jp_out_deg.sort(key=lambda x: -x[1])
    top_jp_citers = []
    for k, deg in jp_out_deg[:15]:
        d = G.nodes[k]
        top_jp_citers.append({
            "pourvoi": k,
            "chamber": d.get("chamber", ""),
            "date": d.get("date", ""),
            "out_degree": deg,
        })

    # Répartition par code
    by_code = Counter(G.nodes[n]["code_slug"] for n in G.nodes
                      if G.nodes[n].get("type") == "article")

    metrics = {
        "name": name,
        "n_nodes": n,
        "n_edges": m,
        "n_jp": n_jp,
        "n_articles": n_art,
        "m_rapproche": m_rapp,
        "m_cite": m_cite,
        "density": nx.density(G),
        "n_components": len(ccs),
        "largest_cc": largest,
        "top30_articles_cited": top_articles,
        "top15_jp_citers": top_jp_citers,
        "by_code_top15": dict(by_code.most_common(15)),
        "compute_seconds": round(time.time() - t0, 1),
    }
    return metrics


# ── Export ─────────────────────────────────────────────────────────────────

def export_graph(G: nx.DiGraph, name: str, include_graphml: bool = True) -> dict:
    sizes = {}
    pkl = OUTPUT_DIR / f"bip-{name}.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(G, f)
    sizes["pickle_mo"] = round(pkl.stat().st_size / 1e6, 2)

    if include_graphml:
        Gx = nx.DiGraph()
        for n, attrs in G.nodes(data=True):
            Gx.add_node(n, **{
                k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                for k, v in attrs.items()
            })
        for u, v, attrs in G.edges(data=True):
            Gx.add_edge(u, v, **{
                k: (v2 if isinstance(v2, (str, int, float, bool)) else str(v2))
                for k, v2 in attrs.items()
            })
        gml = OUTPUT_DIR / f"bip-{name}.graphml"
        nx.write_graphml(Gx, gml)
        sizes["graphml_mo"] = round(gml.stat().st_size / 1e6, 2)
    return sizes


def write_metrics_md(m: dict, sizes: dict) -> Path:
    name = m["name"]
    lines = [
        f"# Graphe bipartite `{name}` — métriques",
        "",
        "## Synthèse",
        "",
        f"- Nœuds : **{m['n_nodes']:,}** ({m['n_jp']:,} JP + {m['n_articles']:,} articles)",
        f"- Arêtes : **{m['n_edges']:,}** ({m['m_rapproche']:,} `rapproche` + {m['m_cite']:,} `cite`)",
        f"- Densité : {m['density']:.2e}",
        f"- Composantes connexes (non dirigé) : {m['n_components']:,} — plus grosse : {m['largest_cc']:,}",
        f"- Temps de calcul : {m['compute_seconds']} s",
        "",
        "## Top 15 codes (par nombre d'articles distincts retenus)",
        "",
    ]
    for code, n in m["by_code_top15"].items():
        lines.append(f"- `{code}` : {n:,}")

    lines += ["", "## Top 30 articles les plus cités dans ce périmètre", "",
              "| pair_key | code | article | in_degree |",
              "|---|---|---|---:|"]
    for a in m["top30_articles_cited"]:
        lines.append(f"| `{a['pair_key']}` | {a['code_slug']} | {a['article_num']} | {a['in_degree']} |")

    lines += ["", "## Top 15 décisions citant le plus d'articles distincts", "",
              "| pourvoi | chambre | date | out_degree (articles cités) |",
              "|---|---|---|---:|"]
    for d in m["top15_jp_citers"]:
        lines.append(f"| {d['pourvoi']} | {d['chamber']} | {d['date']} | {d['out_degree']} |")

    lines += ["", "## Exports", "",
              f"- `bip-{name}.pkl` : {sizes.get('pickle_mo','?')} Mo"]
    if "graphml_mo" in sizes:
        lines.append(f"- `bip-{name}.graphml` : {sizes['graphml_mo']} Mo")

    out = OUTPUT_DIR / f"metrics-{name}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def write_summary(all_metrics: list[dict], scan_stats: dict) -> Path:
    lines = [
        "# Graphes bipartites JP × Articles — synthèse",
        "",
        "Source : `database-judilibre-enrichie/Cour de cassation`.",
        f"Articles uniques dans le corpus : **{scan_stats['pair_keys_unique']:,}** "
        f"(issus de {scan_stats['pairs_seen']:,} citations brutes ; "
        f"{scan_stats['pair_dropped_malformed']} malformées rejetées).",
        "",
        "## Comparatif des 3 périmètres",
        "",
        "| Périmètre | Nœuds | JP | Articles | Arêtes cite | Arêtes rapproche | Composantes | Plus grosse CC | Compute |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in all_metrics:
        lines.append(
            f"| {m['name']} | {m['n_nodes']:,} | {m['n_jp']:,} | {m['n_articles']:,} "
            f"| {m['m_cite']:,} | {m['m_rapproche']:,} | {m['n_components']:,} "
            f"| {m['largest_cc']:,} | {m['compute_seconds']} s |"
        )
    lines += ["",
              "## Format canonique",
              "",
              "- Nœud JP : clé = pourvoi normalisé (`10-87525`). Attributs : type, id_judilibre, ecli, chamber, date, solution.",
              "- Nœud Article : clé = `pair_key` au format `code_slug:article_num` (ex. `code_civil:1240`, `code_du_travail:L122-14-3`).",
              "- Arête `rapproche` (Decision→Decision) : héritée Phase A.",
              "- Arête `cite` (Decision→Article) : extraite de `code_article_pairs`.",
              "",
              "Conforme à [[Format-Fondement-Juridique]] et [[Format-Jurisprudence]].",
              ]
    out = OUTPUT_DIR / "summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Input  : {INPUT_PATH}")
    print(f"Output : {OUTPUT_DIR}")

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        bench = json.load(f)
    benchmark_sources = {
        normalize_pourvoi(q["decision"]["pourvoi"])
        for q in bench["questions"]
        if q["decision"]["pourvoi"]
    }
    print(f"Sources benchmark : {len(benchmark_sources):,}")

    print("\n── Pass 1 : scan du corpus ──────────────────────────────────────")
    data = scan_corpus(INPUT_PATH)

    all_metrics = []
    for periscope, include_graphml in [
        ("resserre", True),
        ("large", True),
        ("tout_cc", False),
    ]:
        print(f"\n── Graphe '{periscope}' ────────────────────────────────")
        G = build_graph(periscope, data, benchmark_sources)
        print(f"  |V|={G.number_of_nodes():,}  |E|={G.number_of_edges():,}")
        m = compute_metrics(G, periscope)
        sizes = export_graph(G, periscope, include_graphml=include_graphml)
        md = write_metrics_md(m, sizes)
        all_metrics.append(m)
        print(f"  ✅ {md.name} ; pickle {sizes.get('pickle_mo','?')} Mo"
              + (f" + graphml {sizes['graphml_mo']} Mo" if 'graphml_mo' in sizes else ""))

    summary = write_summary(all_metrics, data["stats"])
    print(f"\n✅ Synthèse : {summary}")

    print("\n=== Récap bipartite ===")
    for m in all_metrics:
        print(f"  {m['name']:10s} | JP={m['n_jp']:>7,}  Art={m['n_articles']:>6,}  "
              f"cite={m['m_cite']:>7,}  rapp={m['m_rapproche']:>6,}  "
              f"CC largest={m['largest_cc']:>7,}")


if __name__ == "__main__":
    main()
