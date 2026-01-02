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
| [Hybrid BM25 → bi-encoder](hybrid/hybrid_bm25_biencoder.md) (`top_k_candidates=200`, `rerank_k=200`, `alpha=None`) | 0.5050 | 0.2335 | 0.0700 | 0.2804 | 0.1938 |
| [BM25 (Okapi)](lexical/bm25_okapi.md) | 0.4857 | 0.2295 | 0.0689 | 0.2827 | 0.1967 |
| [Hybrid TF-IDF → bi-encoder](hybrid/hybrid_tfidf_biencoder.md) (`top_k_candidates=200`, `rerank_k=200`, `alpha=None`) | 0.4833 | 0.2114 | 0.0645 | 0.2639 | 0.1793 |
| [TF-IDF cosine](lexical/tfidf_cosine.md) | 0.3757 | 0.1613 | 0.0490 | 0.1866 | 0.1278 |

## Notes (verbose)
- All values above come from the *current* cached runs in `.cache/submissions/*/metrics.csv`.
- The hybrid bi-encoder runs use `alpha=None`, which means: rank by the bi-encoder semantic score (cosine similarity of normalized embeddings) and use the lexical score only as a tie-breaker.
- BM25 is a strong lexical baseline for `wiki-trivia` (entity-heavy, Wikipedia-like text), but the BM25→bi-encoder reranker slightly improves Hits@10 while slightly reducing rank-sensitive metrics (MRR/nDCG). This usually indicates “more questions have a hit somewhere in top-10”, but the exact rank of the first relevant hit is not always improved.
- TF-IDF cosine is substantially weaker than BM25 as a pure lexical method here; however, TF-IDF→bi-encoder reranking recovers much of that gap (again with some tradeoff in rank-sensitive metrics).
- Older LSA-based hybrid result pages still exist in `results/hybrid/`, but they do not correspond to any *currently cached* submissions in `.cache/submissions/`.

## Consistency check against submissions
The numbers above match the current cached submissions:
- `.cache/submissions/bm25_okapi/metrics.csv` (run `20260102T172228Z`)
- `.cache/submissions/tfidf_cosine/metrics.csv` (run `20260102T171916Z`)
- `.cache/submissions/hybrid_bm25_biencoder/metrics.csv` (run `20260102T185819Z`)
- `.cache/submissions/hybrid_tfidf_biencoder/metrics.csv` (run `20260102T191425Z`)
