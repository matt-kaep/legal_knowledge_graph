# A3 retrieval depth figure

This two-panel successor replaces the four-panel presentation when the paper needs retrieval depth only. It derives directly from the frozen v3 retrieval tidy CSV; it does not run a model, retrieval method, training, selection, reranking, or E030.

| File | SHA-256 |
|---|---|
| `paper_retrieval_figure.pdf` | `75bc1a99570e66205e0a5788299eeffd3a93d6b0f09ed39da170f5e906066a95` |
| `paper_retrieval_figure.png` | `28901cd69dd737a999ff9571a288db5fe05f14a379ae2392cdd2ee462f9f5ce2` |
| `retrieval_depth_tidy.csv` | `47693fd68e9573470d9edbd2d5e61d70f3b2b4b356011c47a3b4ef945e5d3c4b` |
| `retrieval_figure_manifest.json` | `df84e5f8e05ea41c01240c52a5dcd2de65f8274688d95e4c2cd9897d53b7977f` |

Coverage is 754 A3 evaluation questions, two tasks, three methods, and exact NHit@K for `K=1..100`. Articles use BGE-M3 cosine, PPR G6-AA, and LightGCN G6. Judicial Decisions use BGE-M3 cosine, PPR G7-AA, and LightGCN G6. LightGCN is the frozen mean of seeds 42, 43, and 44.

The y-axis limits are fixed to `0–0.90` for Articles and `0–0.50` for Judicial Decisions to preserve headroom above the highest plotted curve.

Suggested caption: *Retrieval depth on the frozen A3 internal evaluation (754 questions). Curves report exact NHit@K from frozen top-100 rankings for BGE-M3 cosine, PPR, and LightGCN G6. Articles use PPR G6-AA and judicial decisions use PPR G7-AA. Retrieval parameters were selected on grouped training cross-validation before this internal evaluation.*

Status: A3 internal evaluation after train/CV freeze.
