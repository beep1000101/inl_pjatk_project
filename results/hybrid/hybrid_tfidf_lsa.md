# Hybrid TF-IDF → LSA reranking

## Run metadata
- Run ID: `20260102T130439Z`
- Method: `hybrid_tfidf_lsa`
- Dataset: `piotr-rybak__poleval2022-passage-retrieval-dataset`
- Subdataset: `wiki-trivia`
- Questions split: `test`
- Pairs split (labels): `test`
- $k$: `10`
- Questions: `1291` (labeled: `1291`)
- Submission-only: `False`

## Metrics@10
| Metric | Value |
|---|---:|
| Hits@10 | 0.3749 |
| Recall@10 | 0.1603 |
| Precision@10 | 0.0487 |
| MRR@10 | 0.1920 |
| nDCG@10 | 0.1299 |

Source of truth: `.cache/submissions/hybrid_tfidf_lsa/metrics.csv`.

## Output artifacts
- Submission TSV: `.cache/submissions/hybrid_tfidf_lsa/wiki-trivia_questions-test.tsv`
- Metrics CSV: `.cache/submissions/hybrid_tfidf_lsa/metrics.csv`

## Configuration (from metrics.csv)
Two-stage pipeline:
1) Candidate generation: TF‑IDF cosine
2) Reranking: LSA (TruncatedSVD over TF‑IDF) + cosine similarity

Key parameters:
- `top_k_candidates=500`
- `lsa_d=128`
- `rerank_k=100`
- Fusion weight $\alpha=0.9$ with:

$$\text{final} = \alpha\cdot \text{lex} + (1-\alpha)\cdot \text{sem}$$

Other:
- `chunk_size=10000`
- SVD: `svd_n_iter=5`, `svd_random_state=42`

## Interpretation (verbose)
This hybrid is structurally the same as BM25→LSA, but the candidate generator is TF‑IDF instead of BM25. The key practical difference is that the reranker can only help if the relevant passage appears in the candidate set (`top_k_candidates`).

The results are substantially worse than BM25 and also worse than BM25→LSA. The simplest explanation is candidate quality: if TF‑IDF fails to surface the relevant passage in its top-500 candidates for a meaningful fraction of questions, reranking cannot recover those misses. Additionally, because the LSA space is derived from TF‑IDF features, the reranker may not contribute a sufficiently “independent” signal to compensate.

## How to reproduce
`python -m src.eval.hybrid.tfidf_lsa_eval --subdataset wiki-trivia --lsa-d 128 --rerank-k 100 --alpha 0.9`
