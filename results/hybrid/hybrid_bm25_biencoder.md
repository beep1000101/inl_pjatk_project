# Hybrid BM25 → bi-encoder reranking

## Run metadata
- Run ID: `20260102T185819Z`
- Method: `hybrid_bm25_biencoder`
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
| Hits@10 | 0.5050 |
| Recall@10 | 0.2335 |
| Precision@10 | 0.0700 |
| MRR@10 | 0.2804 |
| nDCG@10 | 0.1938 |

Source of truth: `.cache/submissions/hybrid_bm25_biencoder/metrics.csv`.

## Output artifacts
- Submission TSV: `.cache/submissions/hybrid_bm25_biencoder/wiki-trivia_questions-test__20260102T185819Z.tsv`
- Metrics CSV: `.cache/submissions/hybrid_bm25_biencoder/metrics.csv`

## Configuration (from metrics.csv)
Two-stage pipeline:
1) Candidate generation: BM25 (Okapi)
2) Reranking: Sentence-Transformers bi-encoder (cosine similarity)

Key parameters:
- `top_k_candidates=200`
- `rerank_k=200`
- `alpha=None`

Interpretation of `alpha=None`:
- The final ranking is driven by the **semantic** score (bi-encoder cosine similarity) computed between the query and each candidate passage.
- The **lexical** BM25 score is used as a tie-breaker only (i.e., if semantic scores are extremely close).

BM25 parameters:
- `bm25_k1=1.5`, `bm25_b=0.75`

Bi-encoder parameters:
- `biencoder_model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `biencoder_device=cuda`
- `biencoder_batch_size=64`
- `biencoder_max_length=256`

Other:
- `chunk_size=10000`

## Interpretation (verbose)
This method is designed to be “BM25 for recall, bi-encoder for precision (ordering)”. In practice:
- BM25 retrieves a manageable candidate set (`top_k_candidates`), which is crucial given a corpus size of ~6.6M passages.
- The bi-encoder reranker then *re-scores* and re-orders only those candidates.

Compared to the pure BM25 baseline on the same split:
- Hits@10 improves (0.5050 vs 0.4857). This suggests the reranker is often able to pull a relevant passage into the top-10 when BM25 had it “near the top but not quite”.
- MRR@10 and nDCG@10 are slightly lower than BM25 (0.2804/0.1938 vs 0.2827/0.1967). This usually means that while the reranker increases the probability of *any* hit in the top-10, it does not consistently place the first relevant passage at rank 1–3 (or it sometimes demotes a very-high-rank BM25 hit).

Operational notes:
- This run used `cuda`. If CUDA is unavailable (or incompatible), the reranker has a built-in CPU fallback, but it will be much slower.
- With `alpha=None`, this is not “fusion”; it’s effectively a semantic re-ordering of BM25 candidates.

## How to reproduce
`python -m src.eval.hybrid.bm25_biencoder_eval --subdataset wiki-trivia --split test --k 10 --top-k-candidates 200 --rerank-k 200 --device cuda --batch-size 64 --max-length 256`
