# Evaluation (Baselines + shared evaluation helpers)

This folder contains:

1) CLI scripts to evaluate **single-stage lexical baselines** (TF‑IDF, BM25)
2) Shared utilities to compute metrics and write PolEval-format submissions

For each question (query), the goal is to retrieve the associated passage(s) and output an ordered top‑$k$ list of passage IDs (the PolEval submission format uses $k=10$).

While the official PolEval metric is **nDCG@10**, the project also reports a small set of standard IR diagnostics (Hits/Recall/Precision/MRR) to make debugging and comparisons easier.

## Methods

### 1) TF-IDF + cosine similarity

- Script: `src/eval/tfidf_eval.py`
- Representation: sparse TF-IDF vectors over passages and queries.
- Retrieval: cosine similarity (implemented as matrix dot-products; when TF-IDF is L2-normalized, dot-product equals cosine).

### 2) BM25 (Okapi)

- Script: `src/eval/bm25_eval.py`
- Representation: sparse term-frequency (CountVectorizer) over passages and queries.
- Retrieval: Okapi BM25 scoring (k1/b) computed in passage chunks.

BM25 uses an explicit, deterministic tokenizer: lowercased `\w+` tokens.

## Outputs

Both scripts write into per-method directories under:

- `.cache/submissions/<method>/wiki-trivia_questions-<split>.tsv`
- `.cache/submissions/<method>/metrics.csv`

Where `<method>` is the `run_name` used by the script:

- TF-IDF: `tfidf_cosine`
- BM25: `bm25_okapi`

### TSV format

The TSV is written without a header. Each row contains the top-k predicted passage IDs for a single question.

### metrics.csv format

- Appends **one row per run**.
- Columns are stable (metric names + key run metadata).
- Useful for comparing runs across parameters and methods.

## Running

Run commands from the repository root.

### TF-IDF eval

Prerequisite: cached TF-IDF artifacts must exist.

- Build/cache TF-IDF for `wiki-trivia`:
  - `python -m src.preprocess.tf_idf_vectors wiki-trivia`
- Evaluate:
  - `python src/eval/tfidf_eval.py --subdataset wiki-trivia --split test --k 10`

Useful knobs:
- `--chunk-size`: controls RAM usage (larger can be faster but uses more memory).

### BM25 eval

Prerequisite: cached BM25 term-frequency artifacts must exist.

- Build/cache BM25 artifacts for `wiki-trivia`:
  - `python -m src.preprocess.bm25_vectors wiki-trivia`
- Evaluate:
  - `python src/eval/bm25_eval.py --subdataset wiki-trivia --split test --k 10`

Useful knobs:
- `--k1`, `--b`: BM25 parameters.
- `--chunk-size`: RAM/speed tradeoff.

## What does --submission-only do?

Both scripts accept `--submission-only`.

- Still writes the submission TSV.
- Skips loading relevance labels (`pairs-*.tsv`) and therefore **does not compute metrics**.
- The `metrics.csv` row will still be appended, but metric columns will be empty.

Use this when:
- you only need the TSV for submission, or
- labels are unavailable for the chosen split.

## Where are hybrid runs evaluated?

Hybrid (two-stage) systems live under:

- `src/eval/hybrid/`

Those scripts still rely on the shared helpers in this folder:

- `retrieval_eval.py` is the common “load questions → retrieve → write TSV → compute metrics” pipeline.
- `metrics_io.py` is the common “append a stable run row to metrics.csv” logger.

This split is intentional:
- `src/eval/` stays focused on the common evaluation contract and baselines.
- `src/eval/hybrid/` focuses on wiring lexical retrievers + rerankers together.
