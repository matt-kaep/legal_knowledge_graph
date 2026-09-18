# Paper handoff — A3/E029 depth figure audit

The existing figure passed a read-only contract audit. Use the files in the parent directory; this audit did not regenerate the PDF, PNG, or CSVs.

| Deliverable | Path | SHA-256 |
|---|---|---|
| Vector figure | `../paper_depth_figure.pdf` | `e7f77012c544163e35c433c6c381650333744e13ea54561bec7f818cb87b61e2` |
| 320 dpi PNG | `../paper_depth_figure.png` | `b5968ad16af04f68e49edc40751b70be9a9a9c9bd03d827e0bb7a2d6a26ea112` |
| Retrieval tidy data | `../retrieval_depth_tidy.csv` | `47693fd68e9573470d9edbd2d5e61d70f3b2b4b356011c47a3b4ef945e5d3c4b` |
| Reranking tidy data | `../reranking_depth_tidy.csv` | `27731e70636902acb147a9f48b27652191ede3f728a44be4f859e4bdcdaf0162` |
| Existing figure manifest | `../depth_figure_manifest.json` | `3954b06f9d383de21da7788993a6a16a7a68e5e48dad71195375e3466fb02f75` |

The four panels cover 754 frozen A3 questions: Articles and Judicial Decisions retrieval at `K=1..100`, then E029 output-512 reranking at `K_in=10..70` and `K_out=10`. Retrieval uses cosine, PPR, and LightGCN G6. Articles use PPR G6-AA. Judicial Decisions use PPR G7-AA for retrieval, but PPR G6-AA for the distinct E029 reranking experiment. The JP PPR reranking baseline is `0.22822723253757737`, and its `K_in=70` result is `0.308576`.

LightGCN lines are the mean of frozen seeds 42, 43, and 44. Solid lines are reranked top-10 results; dashed lines are retriever-only NHit@10. No Direct LLM curve, uncertainty interval, or error bar appears. The E029 source is `b2_reranking_comparable_a3_k70_article192_output512_execution_v2.json`, SHA-256 `c33f4474aa47224a884eabc7e9f616f7b7b7cd1d6db49113619449dce1522586`; output-256 and E017/E021/E022 are excluded.

Suggested caption: *Retrieval and reranking depth on the frozen A3 internal evaluation (754 questions). Top panels report exact NHit@K from frozen top-100 rankings. Bottom panels report final NHit@10 after E029 reranking at K_out=10; solid curves are reranked results and dashed lines are the corresponding retriever NHit@10. The jurisprudence PPR reranking curve uses G6-AA frozen pools and is distinct from the PPR G7-AA retrieval line in the main table. E029 is exploratory.*

Status: retrieval is A3 internal evaluation after train/CV freeze; E029 remains exploratory.
