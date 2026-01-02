# `src/preprocess/` – Precomputations and caching for fast retrieval

This module exists because the PolEval passage corpus is **huge** (millions of passages). If every evaluation run re-read and re-vectorized `passages.jl`, experimentation would be painfully slow.

So we do what real retrieval systems do: **precompute passage representations once**, write them to disk, and reuse them across runs.

## What problems preprocessing solves

1) **Speed**
- Vectorizing ~7M passages is expensive.
- Caching turns “minutes/hours per run” into “seconds to load artifacts”.

2) **Consistency**
- Candidate indices from lexical retrieval must map to the same passage ordering used by rerankers.
- Preprocessing establishes a stable passage id ordering saved alongside matrices.

3) **Reproducibility**
- Artifacts are stored under `.cache/preprocessed_data/<method>/<subdataset>/…` with a `meta.json` describing parameters.

## What’s inside

### Lexical vector caches

- `tf_idf_vectors.py`
  - Builds TF‑IDF vectors for all passages in a subdataset.
  - Stores:
    - `vectorizer.joblib`
    - `passages_tfidf.npz` (sparse matrix)
    - `passage_ids.npy`
    - `meta.json`

- `bm25_vectors.py`
  - Builds CountVectorizer term-frequency matrix + BM25-specific stats.
  - Stores:
    - `vectorizer.joblib`
    - `passages_tf.npz`
    - `passage_ids.npy`
    - `doc_len.npy`, `idf.npy`
    - `meta.json`

Both modules implement small text hygiene that matters for Wikipedia-like corpora:
- dropping redirect-like passages (`REDIRECT`, `PATRZ`)
- using `title + text` when available

### Semantic / dimensionality reduction caches

- `lsa_svd.py`
  - Trains and caches a TruncatedSVD model on TF‑IDF features.
  - Used by LSA rerankers as a cheap “semantic-ish” signal.

### Model download helpers

- `download_biencoder.py`
  - Convenience helper to download Sentence-Transformers models ahead of time.
  - This reduces “first run” surprises and makes runs more reproducible offline.

## How it helps solve PolEval Task 3

The task requires outputting **top-10 passage IDs** for each question, across multiple domains.

- Lexical retrieval (BM25 / TF‑IDF) is competitive in zero-shot setups and is fast enough to run over large corpora if you have cached matrices.
- Hybrid approaches need lexical candidate generation to avoid embedding *every* passage for *every* query.

## Typical usage

From repo root:

- `python -m src.preprocess.tf_idf_vectors wiki-trivia`
- `python -m src.preprocess.bm25_vectors wiki-trivia`

These commands create the caches used by [src/eval/](../eval/README.md) and [src/hybrid/](../hybrid/README.md).
