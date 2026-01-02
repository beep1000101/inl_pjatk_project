from __future__ import annotations

from pathlib import Path

import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def page() -> None:
    st.title("Problem Summary")

    st.markdown(
        """
## Task framing (as implemented in this repo)

This repository implements **PolEval 2022 Task 3 (Passage Retrieval)**:

- For each question, produce an **ordered list of 10 passage IDs** from the provided corpus.

This Streamlit presentation is **scoped strictly to the `wiki-trivia` subdataset** from the cached dataset snapshot:

- `.cache/data/piotr-rybak__poleval2022-passage-retrieval-dataset/wiki-trivia`

The repository codebase supports cross-domain evaluation, but this app **ignores** all non‑`wiki-trivia` data and results.

## Data and evaluation setup (repo structure)

- Questions:
  - `questions-train.jl`, `questions-test.jl`
- Passage corpus:
  - `passages.jl`
- Labels for evaluation:
  - `pairs-train.tsv`, `pairs-test.tsv`

## Implemented retrieval families

- **Lexical baselines**:
  - TF‑IDF + cosine similarity
  - BM25 (Okapi)

- **Hybrid (lexical candidates + semantic reranking)**:
  - BM25 → bi-encoder reranker
  - TF‑IDF → bi-encoder reranker

All metrics shown in the rest of the app come from experiment logs written as rows in:

- `.cache/submissions/<run_name>/metrics.csv`

(There are no metrics on this page.)
"""
    )

    root = _project_root()
    dataset_dir = root / ".cache" / "data" / "piotr-rybak__poleval2022-passage-retrieval-dataset" / "wiki-trivia"

    st.subheader("Dataset location (wiki-trivia only)")
    st.code(str(dataset_dir))


page()
