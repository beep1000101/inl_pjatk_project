# TF-IDF + Cosine

## Run metadata
- Run ID: `20260102T171916Z`
- Method: `tfidf_cosine`
- Dataset: `piotr-rybak__poleval2022-passage-retrieval-dataset`
- Subdataset: `wiki-trivia`
- Questions split: `test`
 Pairs split (labels): `test`
- $k$: `10`
 Questions: `1291` (labeled: `1291`)
 Submission-only: `False`

## Metrics@10
| Metric | Value |
|---|---:|
| Hits@10 | 0.3757 |
| Recall@10 | 0.1613 |
| Precision@10 | 0.0490 |
| MRR@10 | 0.1866 |
| nDCG@10 | 0.1278 |

## Output artifacts
- Submission TSV: `.cache/submissions/tfidf_cosine/wiki-trivia_questions-test.tsv`
- Metrics CSV: `.cache/submissions/tfidf_cosine/metrics.csv`

## Configuration (from metrics.csv)
- TF-IDF cache: `.cache/preprocessed_data/tf_idf_vectors/wiki-trivia`
- Vectorizer: `min_df=5`, `max_df=0.9`, `max_features=500000`
- Corpus size: `n_passages=6639839`, `n_features=500000`
- Chunking during scoring: `chunk_size=10000`

## Interpretation (verbose)
TF‑IDF retrieval is still lexical: it rewards shared tokens, but uses inverse-document-frequency weighting to down-weight common words. It can sometimes beat raw TF-style baselines when passages are long or when stopword-like terms dominate raw counts.

One practical gotcha (worth calling out because it previously bit us in this repo): if you run with `--submission-only`, the pipeline will intentionally skip metric computation (no label pairs loaded) and you’ll end up with a TSV but blank metrics.

## How to reproduce
- Build cache: `python -m src.preprocess.tf_idf_vectors wiki-trivia`
- Evaluate (with metrics): `python -m src.eval.tfidf_eval --subdataset wiki-trivia --split test --k 10`
