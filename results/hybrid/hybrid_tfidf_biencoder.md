# Hybrid TF-IDF → bi-encoder reranking

## Run metadata
- Run ID: `20260102T191425Z`
- Method: `hybrid_tfidf_biencoder`
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
| Hits@10 | 0.4833 |
| Recall@10 | 0.2114 |
| Precision@10 | 0.0645 |
| MRR@10 | 0.2639 |
| nDCG@10 | 0.1793 |

Source of truth: `.cache/submissions/hybrid_tfidf_biencoder/metrics.csv`.

## Output artifacts
- Submission TSV: `.cache/submissions/hybrid_tfidf_biencoder/wiki-trivia_questions-test__20260102T191425Z.tsv`
- Metrics CSV: `.cache/submissions/hybrid_tfidf_biencoder/metrics.csv`

## Configuration (from metrics.csv)
Two-stage pipeline:
1) Candidate generation: TF-IDF cosine
2) Reranking: Sentence-Transformers bi-encoder (cosine similarity)

Key parameters:
- `top_k_candidates=200`
- `rerank_k=200`
- `alpha=None`

Interpretation of `alpha=None`:
- The final ranking is driven by the **semantic** score (bi-encoder cosine similarity) computed between the query and each candidate passage.
- The TF-IDF lexical score is only used as a tie-breaker.

Bi-encoder parameters:
- `biencoder_model=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- `biencoder_device=cuda`
- `biencoder_batch_size=64`
- `biencoder_max_length=256`

Other:
- `chunk_size=10000`

## Interpretation (verbose)
This hybrid has the same “lexical candidate set + semantic rerank” structure as BM25→bi-encoder, but starts from TF‑IDF instead of BM25.

The main limiting factor for any reranking pipeline is candidate recall:
- A reranker can only promote relevant passages that are already present in the candidate set (`top_k_candidates`).
- If TF‑IDF misses the relevant passage entirely (not in top-200), no amount of semantic reranking can recover it.

That said, this run performs dramatically better than pure TF‑IDF@10 (Hits@10 0.4833 vs 0.3757). That strongly suggests that for a substantial slice of questions:
- TF‑IDF retrieves the relevant passage somewhere in its top-200,
- but does *not* rank it in the top-10,
- and the bi-encoder is able to identify it and push it upward.

Compared to BM25 baseline and BM25→bi-encoder:
- Hits@10 is close to BM25 (0.4833 vs 0.4857), but rank-sensitive metrics remain lower (MRR/nDCG). In other words: this pipeline often finds a hit in top-10, but the first hit tends to appear later in the list than BM25.

Operational notes:
- This run used `cuda`. CPU fallback exists but is significantly slower.
- With `alpha=None`, this is not score fusion; it’s semantic re-ordering of TF‑IDF candidates.

## How to reproduce
`python -m src.eval.hybrid.tfidf_biencoder_eval --subdataset wiki-trivia --split test --k 10 --top-k-candidates 200 --rerank-k 200 --device cuda --batch-size 64 --max-length 256`
