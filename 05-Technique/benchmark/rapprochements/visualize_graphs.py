"""Visualise les 6 graphes produits :
  - Phase A (rapprochements JP-JP)    : resserre / large / tout_cc
  - Phase B (bipartite JP × Articles) : resserre / large / tout_cc

Stratégie adaptée à la taille du graphe :
  - Petit (<= 10k nœuds)    : rendu complet, spring_layout
  - Moyen (10-100k)         : plus grosse composante uniquement, ou k-core
  - Énorme (>100k)          : top-K hubs + leurs voisins immédiats

Couleurs :
  - Phase A : par chambre (Soc./Crim./Civ. 1re/2e/3e/Com./Ass. plén./Ch. mixte/autre)
  - Phase B : JP (bleu nuancé par chambre) + Articles (rouge/orange)

Sortie : data/figures/*.png
"""

from __future__ import annotations

import pickle
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # pas de display
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).parent
GRAPHS_A = ROOT / "data" / "graphs"
GRAPHS_B = ROOT / "data" / "graphs_bipartite"
FIG_DIR = ROOT / "data" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette par chambre ────────────────────────────────────────────────────
CHAMBER_COLORS = {
    "Soc.":        "#1f77b4",  # bleu
    "Crim.":       "#d62728",  # rouge
    "Civ. 1re":    "#2ca02c",  # vert
    "Civ. 2e":     "#9467bd",  # violet
    "Civ. 3e":     "#8c564b",  # marron
    "Com.":        "#ff7f0e",  # orange
    "Ass. plén.":  "#e377c2",  # rose
    "Ch. mixte":   "#17becf",  # cyan
}
DEFAULT_DECISION_COLOR = "#7f7f7f"  # gris
ARTICLE_COLOR = "#ff9896"           # rouge pâle


def load_graph(path: Path) -> nx.DiGraph:
    print(f"  charge {path.name} ...", end=" ", flush=True)
    t0 = time.time()
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"{G.number_of_nodes():,} nodes / {G.number_of_edges():,} edges ({time.time()-t0:.1f}s)")
    return G


# ── Stratégies de sous-échantillonnage ──────────────────────────────────────

def largest_connected_subgraph(G: nx.Graph) -> nx.Graph:
    Gu = G.to_undirected() if G.is_directed() else G
    ccs = list(nx.connected_components(Gu))
    if not ccs:
        return G
    biggest = max(ccs, key=len)
    sub = G.subgraph(biggest).copy()
    return sub


def k_core_subgraph(G: nx.Graph, k: int) -> nx.Graph:
    """Retient les nœuds qui restent après élimination itérative des nœuds
    de degré < k. Multigraph-safe."""
    Gu = G.to_undirected() if G.is_directed() else G
    Gu = nx.Graph(Gu)  # squash multi-edges
    Gu.remove_edges_from(nx.selfloop_edges(Gu))
    core = nx.k_core(Gu, k)
    return core


def top_hub_neighborhood(G: nx.DiGraph, n_hubs: int, neighbor_limit: int,
                         hub_type: str | None = None) -> nx.DiGraph:
    """Extrait les top-N hubs (par degré total) et ajoute leurs voisins directs
    (jusqu'à neighbor_limit par hub)."""
    if hub_type is not None:
        candidates = [n for n, d in G.nodes(data=True) if d.get("type") == hub_type]
    else:
        candidates = list(G.nodes)
    degrees = [(n, G.degree(n)) for n in candidates]
    degrees.sort(key=lambda x: -x[1])
    hubs = [n for n, _ in degrees[:n_hubs]]
    keep = set(hubs)
    for h in hubs:
        neigh = list(G.neighbors(h)) + list(G.predecessors(h))
        keep.update(neigh[:neighbor_limit])
    return G.subgraph(keep).copy()


# ── Rendu ──────────────────────────────────────────────────────────────────

def render_phase_a(G: nx.DiGraph, title: str, out_path: Path,
                   sub_strategy: str = "full", note: str = ""):
    """Rendu Phase A (JP-JP). Couleurs par chambre."""
    t0 = time.time()
    print(f"  [renderA] strat={sub_strategy} ...", end=" ", flush=True)

    # Sous-échantillonnage
    if sub_strategy == "largest_cc":
        G = largest_connected_subgraph(G)
        note = f"plus grosse composante : {G.number_of_nodes():,} nœuds. {note}".strip()
    elif sub_strategy == "k_core_2":
        G = k_core_subgraph(G, 2)
        note = f"k-core 2 : {G.number_of_nodes():,} nœuds. {note}".strip()
    elif sub_strategy == "full":
        pass

    n = G.number_of_nodes()
    if n == 0:
        print("vide — skip.")
        return

    # Layout adaptatif
    k_val = 1 / (n ** 0.5) if n > 0 else None
    iters = 50 if n < 3_000 else (30 if n < 15_000 else 15)
    print(f"layout spring (iter={iters}) ...", end=" ", flush=True)
    pos = nx.spring_layout(G.to_undirected(), seed=42, iterations=iters, k=k_val)

    # Couleurs
    colors = []
    for node in G.nodes:
        ch = G.nodes[node].get("chamber", "")
        colors.append(CHAMBER_COLORS.get(ch, DEFAULT_DECISION_COLOR))

    # Taille des nœuds proportionnelle au degré
    degrees = dict(G.to_undirected().degree())
    sizes = [3 + 2 * degrees.get(n, 0) for n in G.nodes]

    fig, ax = plt.subplots(figsize=(14, 11))
    nx.draw_networkx_edges(G.to_undirected(), pos, alpha=0.15, width=0.3, edge_color="#444", ax=ax)
    nx.draw_networkx_nodes(G.to_undirected(), pos, node_color=colors, node_size=sizes,
                           alpha=0.8, linewidths=0, ax=ax)

    # Légende chambres (seules celles présentes)
    present_chambers = Counter(G.nodes[n].get("chamber", "") for n in G.nodes)
    legend_items = [(ch, c) for ch, c in CHAMBER_COLORS.items() if present_chambers.get(ch, 0) > 0]
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=8, label=f"{ch} ({present_chambers[ch]:,})")
               for ch, c in legend_items]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    subtitle = f"{n:,} nœuds · {G.number_of_edges():,} arêtes"
    if note:
        subtitle += f"\n{note}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ {out_path.name} ({time.time()-t0:.1f}s)")


def render_phase_b(G: nx.DiGraph, title: str, out_path: Path,
                   sub_strategy: str = "full", note: str = ""):
    """Rendu Phase B (bipartite). Couleurs : chambre pour JP, rouge pour Article."""
    t0 = time.time()
    print(f"  [renderB] strat={sub_strategy} ...", end=" ", flush=True)

    if sub_strategy == "largest_cc":
        G = largest_connected_subgraph(G)
        note = f"plus grosse composante : {G.number_of_nodes():,} nœuds. {note}".strip()
    elif sub_strategy == "k_core_2":
        G = k_core_subgraph(G, 2)
        note = f"k-core 2 : {G.number_of_nodes():,} nœuds. {note}".strip()
    elif sub_strategy == "top_hubs":
        G = top_hub_neighborhood(G, n_hubs=50, neighbor_limit=40, hub_type="article")
        note = f"top 50 articles les plus cités + voisinage : {G.number_of_nodes():,} nœuds. {note}".strip()

    n = G.number_of_nodes()
    if n == 0:
        print("vide — skip.")
        return

    k_val = 1 / (n ** 0.5) if n > 0 else None
    iters = 50 if n < 3_000 else (30 if n < 15_000 else 15)
    print(f"layout spring (iter={iters}) ...", end=" ", flush=True)
    pos = nx.spring_layout(G.to_undirected(), seed=42, iterations=iters, k=k_val)

    # Couleurs selon type
    colors, sizes = [], []
    degs = dict(G.to_undirected().degree())
    for node in G.nodes:
        nt = G.nodes[node].get("type", "decision")
        if nt == "article":
            colors.append(ARTICLE_COLOR)
            sizes.append(10 + 1.5 * degs.get(node, 0))
        else:
            ch = G.nodes[node].get("chamber", "")
            colors.append(CHAMBER_COLORS.get(ch, DEFAULT_DECISION_COLOR))
            sizes.append(3 + 1.2 * degs.get(node, 0))

    fig, ax = plt.subplots(figsize=(15, 12))
    nx.draw_networkx_edges(G.to_undirected(), pos, alpha=0.08, width=0.25, edge_color="#666", ax=ax)
    nx.draw_networkx_nodes(G.to_undirected(), pos, node_color=colors, node_size=sizes,
                           alpha=0.85, linewidths=0, ax=ax)

    present_chambers = Counter(G.nodes[n].get("chamber", "")
                               for n in G.nodes if G.nodes[n].get("type") == "decision")
    n_articles = sum(1 for n in G.nodes if G.nodes[n].get("type") == "article")
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c,
                          markersize=8, label=f"{ch} ({present_chambers[ch]:,})")
               for ch, c in CHAMBER_COLORS.items() if present_chambers.get(ch, 0) > 0]
    handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=ARTICLE_COLOR,
                              markersize=10, label=f"Articles ({n_articles:,})"))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)

    subtitle = f"{n:,} nœuds · {G.number_of_edges():,} arêtes"
    if note:
        subtitle += f"\n{note}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"✅ {out_path.name} ({time.time()-t0:.1f}s)")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"Figures → {FIG_DIR}\n")

    print("=== Phase A — Rapprochements JP-JP ===")

    # A.resserre : full
    G = load_graph(GRAPHS_A / "rapp-resserre.pkl")
    render_phase_a(G, "Phase A — resserre : rapprochements (benchmark + cibles)",
                   FIG_DIR / "A1-rapp-resserre.png", sub_strategy="full")

    # A.large : full (21k nœuds, OK avec 15 itérations)
    G = load_graph(GRAPHS_A / "rapp-large.pkl")
    render_phase_a(G, "Phase A — large : tous arrêts avec ≥1 rapprochement",
                   FIG_DIR / "A2-rapp-large.png", sub_strategy="full")

    # A.tout_cc : largest connected component (514k isolés, on garde la structure)
    G = load_graph(GRAPHS_A / "rapp-tout_cc.pkl")
    render_phase_a(G, "Phase A — tout_cc : plus grosse composante du corpus CC",
                   FIG_DIR / "A3-rapp-tout_cc.png", sub_strategy="largest_cc")

    print("\n=== Phase B — Bipartite JP × Articles ===")

    # B.resserre : full
    G = load_graph(GRAPHS_B / "bip-resserre.pkl")
    render_phase_b(G, "Phase B — resserre : JP × Articles (benchmark)",
                   FIG_DIR / "B1-bip-resserre.png", sub_strategy="full")

    # B.large : 31k nœuds, full (15 itérations)
    G = load_graph(GRAPHS_B / "bip-large.pkl")
    render_phase_b(G, "Phase B — large : JP × Articles (tous liens rapprochement)",
                   FIG_DIR / "B2-bip-large.png", sub_strategy="full")

    # B.tout_cc : 568k — trop gros, on prend top 50 hubs articles + voisinage
    G = load_graph(GRAPHS_B / "bip-tout_cc.pkl")
    render_phase_b(G, "Phase B — tout_cc : top 50 articles les plus cités + voisinage JP",
                   FIG_DIR / "B3-bip-tout_cc-hubs.png", sub_strategy="top_hubs")

    print(f"\n✅ 6 figures produites dans {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        mo = f.stat().st_size / 1e6
        print(f"  {f.name}  ({mo:.1f} Mo)")


if __name__ == "__main__":
    main()
