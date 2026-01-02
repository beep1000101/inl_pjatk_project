# Results (wiki-trivia)

## Run metadata (common)
- Dataset: `piotr-rybak__poleval2022-passage-retrieval-dataset`
- Subdataset: `wiki-trivia`
- Questions split: `test`
- $k$: `10`
- Questions: `1291`

Source of truth: `.cache/submissions/*/metrics.csv`.

## Snapshot (metrics@10)
| Method | Hits@10 | Recall@10 | Precision@10 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| [BM25 (Okapi)](lexical/bm25_okapi.md) | 0.4849 | 0.2294 | 0.0689 | 0.2826 | 0.1966 |
| [Hybrid BM25 → LSA](hybrid/hybrid_bm25_lsa.md) (`lsa_d=128`, `rerank_k=100`, $\alpha=0.9$) | 0.4810 | 0.2266 | 0.0680 | 0.2808 | 0.1947 |
| [Hybrid TF-IDF → LSA](hybrid/hybrid_tfidf_lsa.md) (`lsa_d=128`, `rerank_k=100`, $\alpha=0.9$) | 0.3749 | 0.1603 | 0.0487 | 0.1920 | 0.1299 |
| [TF-IDF cosine](lexical/tfidf_cosine.md) | — | — | — | — | — |

## Notes (verbose)
- The `tfidf_cosine` rows are blank because the cached TF‑IDF run(s) were executed with `submission_only=True` (i.e. `--submission-only`), so metrics were intentionally not computed.
- BM25 remains the strongest baseline in this snapshot; on Wikipedia-like text and short queries, lexical overlap is a very strong signal.
- LSA reranking with the shown hyperparameters slightly degrades BM25. This is consistent with a weak/noisy semantic score perturbing the top of an already-good lexical ranking.
- TF‑IDF as a candidate generator is notably weaker than BM25 here, and LSA reranking does not close that gap.

## Consistency check against submissions
The numbers above match the current cached submissions:
- `.cache/submissions/bm25_okapi/metrics.csv` (run `20260102T105510Z`)
- `.cache/submissions/hybrid_bm25_lsa/metrics.csv` (run `20260102T132103Z`)
- `.cache/submissions/hybrid_tfidf_lsa/metrics.csv` (run `20260102T130439Z`)
- `.cache/submissions/tfidf_cosine/metrics.csv` (runs `20260102T101358Z`, `20260102T102235Z`, submission-only)
