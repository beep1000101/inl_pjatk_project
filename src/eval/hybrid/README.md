# `src/eval/hybrid/` – Hybrid evaluation CLIs (two-stage systems)

This module contains command-line scripts that **run and evaluate hybrid pipelines**:

1) lexical candidate generation (BM25 or TF‑IDF)
2) semantic reranking (LSA or Sentence-Transformers bi-encoder)

The scripts here are intentionally thin: they mainly parse arguments, wire components together, and rely on shared evaluation utilities.

## Why a separate `eval/hybrid` package exists

Hybrid evaluation is structurally different from a single-stage lexical baseline:

- you need *two* sets of hyperparameters (candidate size + reranker settings),
- you want to log them all into `metrics.csv` so runs are comparable,
- you want the same TSV-writing and metric computation logic as baselines.

Keeping hybrid CLIs in a separate folder prevents `src/eval/` from becoming a grab bag.

## What scripts exist

### BM25 candidates

- `bm25_lsa_eval.py`
  - Candidate generator: BM25
  - Reranker: LSA (TruncatedSVD over TF‑IDF)

- `bm25_biencoder_eval.py`
  - Candidate generator: BM25
  - Reranker: Sentence-Transformers bi-encoder cosine

### TF‑IDF candidates

- `tfidf_lsa_eval.py`
  - Candidate generator: TF‑IDF
  - Reranker: LSA

- `tfidf_biencoder_eval.py`
  - Candidate generator: TF‑IDF
  - Reranker: Sentence-Transformers bi-encoder cosine

## Inputs and outputs

All scripts:

- load questions from `.cache/data/<dataset-id>/<subdataset>/questions-<split>.jl`
- retrieve and rerank
- write submission TSVs under `.cache/submissions/<method>/…`
- append a run row to `.cache/submissions/<method>/metrics.csv`

When labels are available (i.e., `pairs-<split>.tsv` exists), they compute metrics including **nDCG@10**, which matches the official PolEval scoring target.

## Key knobs you are expected to tune

Hybrid retrieval performance depends strongly on:

- `--top-k-candidates`
  - candidate recall ceiling for the reranker

- `--rerank-k`
  - how many candidates you actually re-embed / re-score (runtime tradeoff)

- `--alpha`
  - if set, fuses lexical and semantic scores; if omitted (`None`), ranks by semantic score only

- model settings (for bi-encoder)
  - `--model-name`, `--device`, `--batch-size`, `--max-length`

## Typical usage

BM25 → bi-encoder (GPU if available):

- `python -m src.eval.hybrid.bm25_biencoder_eval --subdataset wiki-trivia --split test --k 10 --top-k-candidates 200 --rerank-k 200 --device cuda`

TF‑IDF → LSA (fast, CPU-friendly):

- `python -m src.eval.hybrid.tfidf_lsa_eval --subdataset wiki-trivia --split test --k 10 --lsa-d 128 --rerank-k 100 --alpha 0.9`
