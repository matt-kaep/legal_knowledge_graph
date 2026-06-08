"""
12_build_question_enriched_graph.py

Build the tripartite "question-enriched" graph (article x JP x question)
on top of the bipartite article x JP citation graph, then analyse its
structure : gold coverage, connectivity, k-core, target centrality,
question x question induced network, thematic clusters (doc_id).

Outputs:
  - data/question_graph_analysis.json
  - data/fig_question_degree.png
  - data/fig_target_centrality.png
  - data/fig_theme_cohesion.png
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
GRAPH_NPZ = ROOT / "05-Technique/benchmark/baseline_b2/penal_bundle/graph_penal.npz"
JP_INDEX   = ROOT / "05-Technique/benchmark/baseline_b2/penal_bundle/jp_index_penal.parquet"
CORPUS     = ROOT / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/corpus_strict_gemma4-26B-A4B.json"

OUT_DIR = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/.claude/worktrees/etat-lieux-johnny-2026-05-28/05-Technique/benchmark/etape1_embedding_pur/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Load bipartite graph
# ---------------------------------------------------------------------------
print("[1] Loading bipartite article x JP graph...")
npz = np.load(GRAPH_NPZ, allow_pickle=True)
data = npz["data"]
indices = npz["indices"]
indptr = npz["indptr"]
shape = tuple(int(s) for s in npz["shape"])  # (n_jp, n_art)
article_ids = npz["article_ids"]
jp_ids = npz["jp_ids"]

# CSR : rows = JP, cols = articles
M = csr_matrix((data, indices, indptr), shape=shape)
n_jp, n_art = shape
print(f"   n_jp={n_jp}, n_art={n_art}, n_edges={M.nnz}")

art_id_to_idx = {a: i for i, a in enumerate(article_ids)}
jp_id_to_idx  = {j: i for i, j in enumerate(jp_ids)}


# ---------------------------------------------------------------------------
# 2. Load corpus and JP index (pourvoi -> jp_id)
# ---------------------------------------------------------------------------
print("[2] Loading corpus + jp_index...")
with open(CORPUS, "r", encoding="utf-8") as f:
    corpus = json.load(f)
questions = corpus["questions"]
print(f"   n_questions = {len(questions)}")

jp_df = pd.read_parquet(JP_INDEX)
# Build pourvoi -> jp_id only for CC
cc = jp_df[jp_df["juris"] == "CC"]
pourvoi_to_jpid = dict(zip(cc["number"].astype(str), cc["id"].astype(str)))
print(f"   CC pourvois indexed: {len(pourvoi_to_jpid)}")


# ---------------------------------------------------------------------------
# 3. Resolve gold articles / gold JP for each question
# ---------------------------------------------------------------------------
print("[3] Resolving gold articles + JP...")

n_art_gold_total = 0
n_art_gold_resolved = 0
n_jp_gold_total = 0
n_jp_gold_resolved = 0
n_q_with_any_art = 0
n_q_with_any_jp = 0

q_records = []  # list of dicts: qid, doc_id, art_idx_set, jp_idx_set

for q in questions:
    art_idx = set()
    for a in q.get("articles_attendus", []) or []:
        n_art_gold_total += 1
        key = f"{a['code_slug']}:{a['article_num']}"
        if key in art_id_to_idx:
            art_idx.add(art_id_to_idx[key])
            n_art_gold_resolved += 1

    jp_idx = set()
    for j in q.get("jp_attendues", []) or []:
        n_jp_gold_total += 1
        pourvoi = j.get("pourvoi")
        if pourvoi and pourvoi in pourvoi_to_jpid:
            jpid = pourvoi_to_jpid[pourvoi]
            if jpid in jp_id_to_idx:
                jp_idx.add(jp_id_to_idx[jpid])
                n_jp_gold_resolved += 1

    if art_idx:
        n_q_with_any_art += 1
    if jp_idx:
        n_q_with_any_jp += 1

    q_records.append({
        "qid": q["qid"],
        "doc_id": q.get("doc_id"),
        "theme": q.get("theme"),
        "art_idx": art_idx,
        "jp_idx": jp_idx,
        "n_gold_art": len(art_idx),
        "n_gold_jp": len(jp_idx),
    })

print(f"   articles gold: {n_art_gold_resolved}/{n_art_gold_total} "
      f"({100*n_art_gold_resolved/max(n_art_gold_total,1):.1f}%)")
print(f"   JP gold     : {n_jp_gold_resolved}/{n_jp_gold_total} "
      f"({100*n_jp_gold_resolved/max(n_jp_gold_total,1):.1f}%)")
print(f"   q with >=1 art gold resolved: {n_q_with_any_art}/{len(questions)}")
print(f"   q with >=1 JP  gold resolved: {n_q_with_any_jp}/{len(questions)}")


# ---------------------------------------------------------------------------
# 4. Build tripartite graph in networkx
# ---------------------------------------------------------------------------
print("[4] Building tripartite networkx graph...")
G = nx.Graph()

# We add article + JP nodes lazily (only those that appear in edges) to save RAM.
# We add ALL article nodes that are gold targets, plus we add edges JP-article
# from the bipartite matrix for k-core/connectivity restricted to a working set.

# To compute connectivity in reasonable time, we build the full bipartite as edges.
# 642450 edges (data nnz) ; M is binary so edges = nnz.
M_coo = M.tocoo()
print(f"   adding {M_coo.nnz} JP-article edges...")

article_nodes = [f"A:{i}" for i in range(n_art)]
jp_nodes      = [f"J:{i}" for i in range(n_jp)]
G.add_nodes_from(article_nodes, kind="article")
G.add_nodes_from(jp_nodes, kind="jp")

# Add edges in chunks
rows = M_coo.row
cols = M_coo.col
edges_iter = ((f"J:{r}", f"A:{c}") for r, c in zip(rows.tolist(), cols.tolist()))
G.add_edges_from(edges_iter)
print(f"   bipartite edges added : {G.number_of_edges()}")

# Add question nodes + question->article + question->jp edges
n_q_isolated = 0
for rec in q_records:
    qnode = f"Q:{rec['qid']}"
    G.add_node(qnode, kind="question", doc_id=rec["doc_id"])
    has_any = False
    for ai in rec["art_idx"]:
        G.add_edge(qnode, f"A:{ai}", kind="gold_art")
        has_any = True
    for ji in rec["jp_idx"]:
        G.add_edge(qnode, f"J:{ji}", kind="gold_jp")
        has_any = True
    if not has_any:
        n_q_isolated += 1

print(f"   total nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
print(f"   questions isolated (no resolved neighbor): {n_q_isolated}")


# ---------------------------------------------------------------------------
# 5. Connectivity
# ---------------------------------------------------------------------------
print("[5] Connectivity analysis...")
components = list(nx.connected_components(G))
components.sort(key=len, reverse=True)
n_comp = len(components)
size_main = len(components[0])
print(f"   n_components={n_comp}, main={size_main} ({100*size_main/G.number_of_nodes():.2f}%)")

main_set = components[0]
n_q_in_main = sum(1 for r in q_records if f"Q:{r['qid']}" in main_set)
print(f"   questions in main component: {n_q_in_main}/{len(q_records)}")


# ---------------------------------------------------------------------------
# 6. k-core of articles + JP cited by questions
# ---------------------------------------------------------------------------
print("[6] k-core of question targets...")
# core number on the bipartite (article + JP) subgraph only -- exclude question nodes
# to measure structural centrality of the targets, not inflated by the questions.
Gbp = G.subgraph(article_nodes + jp_nodes).copy()
core = nx.core_number(Gbp)

# stats per question : mean / max core of its gold targets
q_mean_core = []
q_max_core  = []
for rec in q_records:
    targets = [f"A:{i}" for i in rec["art_idx"]] + [f"J:{i}" for i in rec["jp_idx"]]
    cores = [core.get(t, 0) for t in targets]
    if cores:
        q_mean_core.append(float(np.mean(cores)))
        q_max_core.append(int(np.max(cores)))
    else:
        q_mean_core.append(0.0)
        q_max_core.append(0)

print(f"   median mean-core targets : {np.median(q_mean_core):.1f}")
print(f"   median max-core  targets : {np.median(q_max_core):.1f}")


# ---------------------------------------------------------------------------
# 7. Centrality of articles targeted: citations per article in pool
# ---------------------------------------------------------------------------
print("[7] Centrality of targeted articles (citations)...")
art_citations = np.asarray(M.sum(axis=0)).ravel()  # nb of JP citing each article
print(f"   max citations : {art_citations.max()}, mean={art_citations.mean():.1f}")

# For each question, distribution of citation counts of its gold articles
q_target_cit_mean = []
for rec in q_records:
    if rec["art_idx"]:
        c = [int(art_citations[i]) for i in rec["art_idx"]]
        q_target_cit_mean.append(float(np.mean(c)))
    else:
        q_target_cit_mean.append(0.0)

# Buckets
buckets = {"<10": 0, "10-100": 0, "100-1000": 0, ">1000": 0}
for v in q_target_cit_mean:
    if v == 0:
        continue
    if v < 10: buckets["<10"] += 1
    elif v < 100: buckets["10-100"] += 1
    elif v < 1000: buckets["100-1000"] += 1
    else: buckets[">1000"] += 1
print(f"   buckets citation: {buckets}")


# ---------------------------------------------------------------------------
# 8. Question <-> question induced network via shared articles
# ---------------------------------------------------------------------------
print("[8] Question <-> question induced network...")
art_to_qs = defaultdict(list)
for idx, rec in enumerate(q_records):
    for ai in rec["art_idx"]:
        art_to_qs[ai].append(idx)

pair_counts = Counter()
for ai, qs in art_to_qs.items():
    if len(qs) < 2:
        continue
    # share count between every pair
    for i in range(len(qs)):
        for j in range(i + 1, len(qs)):
            a, b = qs[i], qs[j]
            if a > b: a, b = b, a
            pair_counts[(a, b)] += 1

n_pairs = len(pair_counts)
print(f"   n_pairs (q,q) sharing >=1 article : {n_pairs}")

# Build the question-question graph
Hqq = nx.Graph()
Hqq.add_nodes_from(range(len(q_records)))
for (a, b), w in pair_counts.items():
    Hqq.add_edge(a, b, weight=w)

qq_components = list(nx.connected_components(Hqq))
qq_components.sort(key=len, reverse=True)
qq_n_comp = len(qq_components)
qq_main = len(qq_components[0]) if qq_components else 0
qq_isolated = sum(1 for c in qq_components if len(c) == 1)
print(f"   QQ : components={qq_n_comp}, main={qq_main}, isolated_q={qq_isolated}")

# Cliques on QQ (capped, can be expensive)
try:
    cliques = list(nx.find_cliques(Hqq))
    clique_sizes = Counter(len(c) for c in cliques)
    top_clique = max(clique_sizes) if clique_sizes else 0
    n_cliques_ge3 = sum(v for k, v in clique_sizes.items() if k >= 3)
except Exception as e:
    print(f"   clique enum failed: {e}")
    top_clique = -1
    n_cliques_ge3 = -1
    clique_sizes = {}
print(f"   top_clique={top_clique}, n_cliques>=3={n_cliques_ge3}")


# ---------------------------------------------------------------------------
# 9. Thematic clusters : doc_id cohesion
# ---------------------------------------------------------------------------
print("[9] doc_id cohesion...")
doc_to_q_idx = defaultdict(list)
for i, rec in enumerate(q_records):
    if rec["doc_id"]:
        doc_to_q_idx[rec["doc_id"]].append(i)

cohesions = []  # (doc_id, n_q, cohesion_share)
for doc_id, idxs in doc_to_q_idx.items():
    if len(idxs) < 2:
        continue
    n_pairs_doc = len(idxs) * (len(idxs) - 1) // 2
    n_pairs_linked = 0
    idx_set = set(idxs)
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            a, b = idxs[i], idxs[j]
            if a > b: a, b = b, a
            if (a, b) in pair_counts:
                n_pairs_linked += 1
    cohesions.append((doc_id, len(idxs), n_pairs_linked / n_pairs_doc))

cohesions.sort(key=lambda x: -x[2])
mean_cohesion = float(np.mean([c[2] for c in cohesions])) if cohesions else 0.0
print(f"   n_docs (>=2 q): {len(cohesions)}, mean cohesion={mean_cohesion:.3f}")
print(f"   top 5: {cohesions[:5]}")
print(f"   bottom 5: {cohesions[-5:]}")


# ---------------------------------------------------------------------------
# 10. Figures
# ---------------------------------------------------------------------------
print("[10] Generating figures...")

# Fig 1 : distribution degree of questions (n_gold_art, n_gold_jp)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist([r["n_gold_art"] for r in q_records], bins=range(0, 12), color="#355c7d", edgecolor="white")
axes[0].set_title("Nb articles gold par question")
axes[0].set_xlabel("n_gold_art")
axes[0].set_ylabel("n_questions")
axes[1].hist([r["n_gold_jp"] for r in q_records], bins=range(0, 12), color="#8b5a2b", edgecolor="white")
axes[1].set_title("Nb JP gold (résolues) par question")
axes[1].set_xlabel("n_gold_jp_resolved")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_question_degree.png", dpi=130)
plt.close(fig)

# Fig 2 : distribution log of citation counts of targeted articles
fig, ax = plt.subplots(figsize=(8, 4.5))
vals = [v for v in q_target_cit_mean if v > 0]
ax.hist(np.log10(vals), bins=30, color="#355c7d", edgecolor="white")
ax.set_title("Centralité moyenne (log10 nb citations) des articles gold")
ax.set_xlabel("log10(nb JP citant l'article gold) — moyenne par question")
ax.set_ylabel("n_questions")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_target_centrality.png", dpi=130)
plt.close(fig)

# Fig 3 : distribution cohesion par doc_id + scatter (n_q, cohesion)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist([c[2] for c in cohesions], bins=20, color="#2d5d3a", edgecolor="white")
axes[0].set_title("Cohésion par doc_id (% paires liées via article partagé)")
axes[0].set_xlabel("cohesion")
axes[0].set_ylabel("n_docs")
axes[1].scatter([c[1] for c in cohesions], [c[2] for c in cohesions],
                alpha=0.5, color="#355c7d")
axes[1].set_xscale("log")
axes[1].set_xlabel("n_questions dans le doc (log)")
axes[1].set_ylabel("cohesion")
axes[1].set_title("Cohesion vs taille du doc")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_theme_cohesion.png", dpi=130)
plt.close(fig)


# ---------------------------------------------------------------------------
# 11. Save analysis JSON
# ---------------------------------------------------------------------------
print("[11] Saving JSON...")

# Node and edge counts by type
n_q = len(q_records)
n_edges_q_art = sum(r["n_gold_art"] for r in q_records)
n_edges_q_jp = sum(r["n_gold_jp"] for r in q_records)

result = {
    "graph": {
        "nodes": {"article": n_art, "jp": n_jp, "question": n_q},
        "edges": {"jp_cite_article": int(M.nnz),
                  "question_gold_article": n_edges_q_art,
                  "question_gold_jp": n_edges_q_jp},
    },
    "gold_coverage": {
        "n_articles_gold_total": n_art_gold_total,
        "n_articles_gold_resolved": n_art_gold_resolved,
        "pct_articles_resolved": round(100 * n_art_gold_resolved / max(n_art_gold_total, 1), 2),
        "n_jp_gold_total": n_jp_gold_total,
        "n_jp_gold_resolved": n_jp_gold_resolved,
        "pct_jp_resolved": round(100 * n_jp_gold_resolved / max(n_jp_gold_total, 1), 2),
        "n_q_with_any_article": n_q_with_any_art,
        "n_q_with_any_jp": n_q_with_any_jp,
    },
    "connectivity": {
        "n_components_total_graph": n_comp,
        "main_component_size": size_main,
        "main_component_pct": round(100 * size_main / G.number_of_nodes(), 2),
        "n_questions_in_main": n_q_in_main,
        "n_questions_isolated_no_neighbor": n_q_isolated,
    },
    "kcore_targets": {
        "median_mean_core": float(np.median(q_mean_core)),
        "median_max_core": float(np.median(q_max_core)),
        "p25_mean_core": float(np.percentile(q_mean_core, 25)),
        "p75_mean_core": float(np.percentile(q_mean_core, 75)),
        "max_observed_core": int(max(core.values()) if core else 0),
    },
    "target_centrality": {
        "buckets_mean_citation_targeted_articles": buckets,
        "median_mean_citation": float(np.median([v for v in q_target_cit_mean if v > 0])),
    },
    "question_question_network": {
        "n_pairs_sharing_article": n_pairs,
        "n_components": qq_n_comp,
        "main_component_size": qq_main,
        "n_isolated_questions": qq_isolated,
        "top_clique_size": top_clique,
        "n_cliques_ge3": n_cliques_ge3,
    },
    "doc_id_cohesion": {
        "n_docs_with_ge2_questions": len(cohesions),
        "mean_cohesion": mean_cohesion,
        "median_cohesion": float(np.median([c[2] for c in cohesions])) if cohesions else 0.0,
        "top5": [{"doc_id": d, "n_q": n, "cohesion": round(c, 3)} for d, n, c in cohesions[:5]],
        "bottom5": [{"doc_id": d, "n_q": n, "cohesion": round(c, 3)} for d, n, c in cohesions[-5:]],
    },
}

with open(OUT_DIR / "question_graph_analysis.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("DONE.")
print(f"   -> {OUT_DIR / 'question_graph_analysis.json'}")
