# `src/` – Project code overview (PolEval 2022 Task 3: Passage Retrieval)

This repository implements a **retrieval system** for PolEval 2022 Task 3 (Passage Retrieval): for each question, return an **ordered list of 10 passage IDs** from the provided corpus.

The challenge is **cross-domain** (trivia / legal / customer support) and largely **zero-shot** (no training pairs for some domains). That is why the codebase is organized around:

- strong, training-free **lexical baselines** (TF‑IDF / BM25),
- optional **hybrid reranking** (lexical candidates + semantic reranker),
- heavy use of **caching** to make experiments feasible on millions of passages.

The central idea: *separate the pipeline into clear stages, cache the expensive artifacts, and keep evaluation reproducible.*

## How the pieces fit together

A typical end-to-end run looks like:

1) **Get the dataset into the local cache**
- [src/data_source/](data_source/README.md) downloads the dataset repository into `.cache/data/…`.

2) **Precompute passage representations** (so we don’t re-vectorize 7M passages for every run)
- [src/preprocess/](preprocess/README.md) builds and stores passage matrices:
  - TF‑IDF matrix for cosine retrieval
  - BM25 term-frequency matrix + idf/doc_len
  - optional LSA (TruncatedSVD) artifacts

3) **Retrieve / rerank and write a submission TSV**
- [src/eval/](eval/README.md) contains CLI scripts that:
  - load questions, run retrieval, write the TSV in PolEval format,
  - optionally compute metrics when `pairs-*.tsv` labels are available.

4) **Optional: calibrate hyperparameters** (choose good `k`/candidate sizes)
- [src/calibration/](calibration/README.md) helps answer questions like “what `k` gives ≥90% of the final Hits@k?”

## Design principles (why modules are separated)

- **Reproducibility over cleverness**: evaluation scripts write `metrics.csv` rows with run metadata so results can be compared later.
- **Caching as a first-class feature**: the corpus is large enough that repeated full vectorization dominates runtime.
- **Modularity**: retrieval (lexical) and reranking (semantic) are split so you can swap components without rewriting evaluation.
- **Cross-domain reality**: lexical methods remain competitive in zero-shot settings; hybrids exist to explore semantic gains when possible.

## Where outputs live

- Dataset snapshot: `.cache/data/<dataset-id>/…` (downloaded once)
- Preprocessed matrices: `.cache/preprocessed_data/<method>/<subdataset>/…`
- Submission TSVs + metrics: `.cache/submissions/<run_name>/…`

If you’re trying to understand “what happens when I run a script”, start with:
- [src/eval/retrieval_eval.py](eval/retrieval_eval.py) (common evaluation + TSV writing)
- [src/config/paths.py](config/paths.py) (all canonical paths)
