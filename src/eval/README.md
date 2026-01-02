# Evaluation (Baselines)

This folder contains CLI scripts to evaluate two retrieval baselines on the POLEVAL2022 passage-retrieval dataset (notably the `wiki-trivia` subdataset).

For each question (query), the goal is to retrieve the associated passage(s) and rank them.

## Methods

### 1) TF-IDF + cosine similarity

- Script: `src/eval/tfidf_eval.py`
- Representation: sparse TF-IDF vectors over passages and queries.
- Retrieval: cosine similarity (implemented as matrix dot-products; when TF-IDF is L2-normalized, dot-product equals cosine).

### 2) fastText embeddings + FAISS

- Script: `src/eval/ft_eval.py`
- Representation: dense embeddings computed as an average of fastText word vectors.
- Retrieval: FAISS ANN/exact search over passage embeddings.
  - `flat` index: exact search (high RAM).
  - `ivfpq` index: compressed approximate search (lower RAM, requires training; tune `nprobe`).

## Outputs

Both scripts write into per-method directories under:

- `.cache/submissions/<method>/wiki-trivia_questions-<split>.tsv`
- `.cache/submissions/<method>/metrics.csv`

Where `<method>` is the `run_name` used by the script:

- TF-IDF: `tfidf_cosine`
- fastText: `fasttext_faiss`

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

### fastText eval

Prerequisite: cached passage vectors must exist.

- Build/cache fastText passage vectors:
  - `python -m src.preprocess.fasttext_vectors wiki-trivia`
- Evaluate (exact FAISS):
  - `python src/eval/ft_eval.py --subdataset wiki-trivia --split test --k 10 --index-type flat`

Useful knobs:
- `--index-type`: `flat` (exact) or `ivfpq` (compressed approximate).
- `--nlist --m --nbits --train-size --nprobe`: IVF-PQ parameters.
- `--index-path`: optional; if provided, the FAISS index will be saved/loaded from disk.

## What does --submission-only do?

Both scripts accept `--submission-only`.

- Still writes the submission TSV.
- Skips loading relevance labels (`pairs-*.tsv`) and therefore **does not compute metrics**.
- The `metrics.csv` row will still be appended, but metric columns will be empty.

Use this when:
- you only need the TSV for submission, or
- labels are unavailable for the chosen split.
