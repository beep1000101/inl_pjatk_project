# Hybrid BM25 → LSA reranking

## Run metadata
- Run ID: `20260102T132103Z`
- Method: `hybrid_bm25_lsa`
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
| Hits@10 | 0.4810 |
| Recall@10 | 0.2266 |
| Precision@10 | 0.0680 |
| MRR@10 | 0.2808 |
| nDCG@10 | 0.1947 |

Source of truth: `.cache/submissions/hybrid_bm25_lsa/metrics.csv`.

## Output artifacts
- Submission TSV: `.cache/submissions/hybrid_bm25_lsa/wiki-trivia_questions-test.tsv`
- Metrics CSV: `.cache/submissions/hybrid_bm25_lsa/metrics.csv`

## Configuration (from metrics.csv)
Two-stage pipeline:
1) Candidate generation: BM25
2) Reranking: LSA (TruncatedSVD over TF‑IDF) + cosine similarity

Key parameters:
- `top_k_candidates=500`
- `lsa_d=128`
- `rerank_k=100`
- Fusion weight $\alpha=0.9$ with:

$$\text{final} = \alpha\cdot \text{lex} + (1-\alpha)\cdot \text{sem}$$

Other:
- `chunk_size=10000`
- BM25 params: `bm25_k1=1.5`, `bm25_b=0.75`
- SVD: `svd_n_iter=5`, `svd_random_state=42`

## Interpretation (verbose)
This hybrid keeps BM25 as a strong lexical candidate generator, then uses an LSA space to reshuffle the top `rerank_k` candidates. The motivation is that LSA can capture co-occurrence/topic structure and thus promote semantically related passages even if exact tokens differ.

In this specific run, metrics are **slightly worse than pure BM25** across the board. Given $\alpha=0.9$, the semantic score still influences the final ordering; if that semantic signal is noisy or too coarse, it can perturb otherwise-correct BM25 top-10 rankings.

Practical takeaways:
- This configuration is “close” to BM25 but not strictly identical (because $\alpha\ne 1$).
- Increasing `rerank_k` can change which items are affected by semantic reshuffling.
- Sweeping $\alpha$ can reveal whether the semantic score ever helps on this dataset.

## How to reproduce
`python -m src.eval.hybrid.bm25_lsa_eval --subdataset wiki-trivia --lsa-d 128 --rerank-k 100 --alpha 0.9`
