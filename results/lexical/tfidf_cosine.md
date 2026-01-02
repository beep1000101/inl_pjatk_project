# TF-IDF + Cosine

## Run metadata
- Run ID: `20260102T101358Z` (also present: `20260102T102235Z`, same settings)
- Method: `tfidf_cosine`
- Dataset: `piotr-rybak__poleval2022-passage-retrieval-dataset`
- Subdataset: `wiki-trivia`
- Questions split: `test`
- Pairs split (labels): *(missing)*
- $k$: `10`
- Questions: `1291` (labeled: *(not computed)*)
- Submission-only: `True`

## Metrics@10
| Metric | Value |
|---|---:|
| Hits@10 | — |
| Recall@10 | — |
| Precision@10 | — |
| MRR@10 | — |
| nDCG@10 | — |

Why metrics are missing: the cached runs were executed with `submission_only=True` (equivalent to `--submission-only`), so the pipeline wrote the submission TSV but did not load label pairs / compute metrics. This is consistent with `pairs_split` being empty in `.cache/submissions/tfidf_cosine/metrics.csv`.

## Output artifacts
- Submission TSV: `.cache/submissions/tfidf_cosine/wiki-trivia_questions-test.tsv`
- Metrics CSV: `.cache/submissions/tfidf_cosine/metrics.csv`

## Configuration (from metrics.csv)
- TF-IDF cache: `.cache/preprocessed_data/tf_idf_vectors/wiki-trivia`
- Vectorizer: `min_df=5`, `max_df=0.9`, `max_features=500000`
- Chunking during scoring: `chunk_size=10000`

## Interpretation (verbose)
TF‑IDF retrieval is still lexical: it rewards shared tokens, but uses inverse-document-frequency weighting to down-weight common words. It can sometimes beat raw TF-style baselines when passages are long or when stopword-like terms dominate raw counts.

However, this specific cached run is **not comparable** to the other reports in `results/` because it did not compute metrics. If you want this method in the same evaluation table, rerun with labels enabled (i.e., without `--submission-only`, and with an available `pairs_split`).

## How to reproduce
- Build cache: `python -m src.preprocess.tf_idf_vectors wiki-trivia`
- Evaluate (with metrics): `python -m src.eval.tfidf_eval --subdataset wiki-trivia --split test --k 10`
