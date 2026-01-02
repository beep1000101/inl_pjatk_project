# BM25 (Okapi)

## Run metadata
- Run ID: `20260102T172228Z`
- Method: `bm25_okapi`
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
| Hits@10 | 0.4857 |
| Recall@10 | 0.2295 |
| Precision@10 | 0.0689 |
| MRR@10 | 0.2827 |
| nDCG@10 | 0.1967 |

Source of truth: `.cache/submissions/bm25_okapi/metrics.csv`.

## Output artifacts
- Submission TSV: `.cache/submissions/bm25_okapi/wiki-trivia_questions-test.tsv`
- Metrics CSV: `.cache/submissions/bm25_okapi/metrics.csv`

## Configuration (from metrics.csv)
- `bm25_k1=1.5`, `bm25_b=0.75`
- Passage matrix cache: `.cache/preprocessed_data/bm25_vectors/wiki-trivia`
- Vectorizer: `CountVectorizer(token_pattern=\b\w+\b, lowercase=True)`, `min_df=5`, `max_df=0.9`, `max_features=500000`
- Corpus size: `n_passages=6639839`, `n_features=500000`, `avgdl=48.74`
- Chunking during scoring: `chunk_size=10000`

## Interpretation (verbose)
BM25 is a classic lexical retrieval method: it scores passages by term overlap, with (1) saturation of term frequency (`k1`) and (2) document-length normalization (`b`). On Wikipedia-like corpora and short, entity-heavy questions (which `wiki-trivia` resembles), exact matching is often a dominant signal, so BM25 tends to be a strong baseline.

The current run shows solid top-10 behavior across all metrics. Importantly, because this run has `pairs_split=test` and `submission_only=False`, the reported metrics are directly comparable to other non-submission-only runs on the same split.

## How to reproduce
- Build cache: `python -m src.preprocess.bm25_vectors wiki-trivia`
- Evaluate: `python -m src.eval.bm25_eval --subdataset wiki-trivia --split test --k 10`
