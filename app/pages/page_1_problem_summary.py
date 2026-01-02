from __future__ import annotations

from pathlib import Path

import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def page() -> None:
    st.title("Problem Summary")

    st.markdown(
        r"""
## Task framing (as implemented in this repo)

This repository implements **PolEval 2022 Task 3 (Passage Retrieval)**:

- For each question, produce an **ordered list of 10 passage IDs** from the provided corpus.

Why this framing:
- The evaluation and submission contract is stable (always produce top-$k$, typically $k=10$), which makes it
  easy to compare methods run-to-run.
- The repository is organized to keep this contract consistent across baselines and hybrid systems.

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

## Metrics used in this app (simple definitions)

All reported metrics are computed **at cutoff $k$** (typically $k=10$).

Important implementation details (as implemented in the repo):
- Relevance is **binary** from `pairs-*.tsv`. If a `score` column exists, only rows with `score > 0` count as relevant.
- Metrics are computed only for **labeled questions** (questions that have at least one relevant passage in the labels), and then averaged.

For a given question:
- you have a set of relevant passages $R_q$,
- and your model produces a ranked list; consider only the top-$k$ results.

The metrics are:

- **Hits@k** (`hits_at_k`): did we get *at least one* relevant passage anywhere in the top-$k$?
  - per question: 1 (yes) / 0 (no)
  - overall: average of those 0/1 values across labeled questions

- **Recall@k** (`recall_at_k`): what fraction of all relevant passages for that question are included in the top-$k$?
  - per question: `(# relevant in top-k) / (# relevant total)`

- **Precision@k** (`precision_at_k`): what fraction of the top-$k$ list is relevant?
  - per question: `(# relevant in top-k) / k`

- **MRR@k** (`mrr_at_k`): how early is the *first* relevant hit in the ranking?
  - per question: if the first relevant passage is at rank `r` (1..k), score is `1/r`; if no relevant in top-k, score is 0

- **nDCG@k** (`ndcg_at_k`): like precision/recall, but **rank-sensitive** (relevant hits at rank 1 matter more than at rank 10).
  - this repo uses **binary gains** (relevant=1, not relevant=0)
  - and then normalizes by the best possible (ideal) ranking for that question

Sources (implementation):
- `src/eval/retrieval_eval.py` (`compute_metrics_at_k`, `load_relevance_pairs`)

## Why the repo is structured this way

The project deliberately separates concerns (as described in the repository docs):

- **Data access lives in `src/data_source/`** so runs are reproducible and don’t depend on manual downloads.
  The dataset is always placed under `.cache/data/...`.
- **Preprocessing lives in `src/preprocess/`** because the corpus is very large; caching passage
  representations turns repeated experiments from “re-vectorize everything” into “load artifacts”.
- **Evaluation lives in `src/eval/`** and logs stable rows into `metrics.csv` so results can be compared later.
- **Hybrid wiring lives in `src/eval/hybrid/`** so baselines and shared evaluation helpers remain simple.

Sources (repo docs):
- `src/README.md`
- `src/data_source/README.md`
- `src/preprocess/README.md`
- `src/eval/README.md`
- `src/config/README.md`
"""
    )

    with st.expander("Optional: formal formulas"):
      st.markdown("Symbols: $q$ = question, $k$ = cutoff, $R_q$ = relevant set, $P_q^{(k)}$ = top-$k$ predictions.")

      st.markdown("**Hits@k** (`hits_at_k`) — any relevant in top-k")
      st.latex(r"\mathrm{Hits@k}(q) = \mathbb{1}[|R_q \cap P_q^{(k)}| > 0]")

      st.markdown("**Recall@k** (`recall_at_k`) — fraction of relevant recovered")
      st.latex(r"\mathrm{Recall@k}(q) = \frac{|R_q \cap P_q^{(k)}|}{|R_q|}")

      st.markdown("**Precision@k** (`precision_at_k`) — fraction of top-k that is relevant")
      st.latex(r"\mathrm{Precision@k}(q) = \frac{|R_q \cap P_q^{(k)}|}{k}")

      st.markdown("**MRR@k** (`mrr_at_k`) — reciprocal rank of first relevant (within top-k)")
      st.latex(
        r"\mathrm{MRR@k}(q) = \begin{cases}1/r_q & \text{if first relevant is at rank } r_q \le k \\ 0 & \text{otherwise}\end{cases}"
      )

      st.markdown("**nDCG@k** (`ndcg_at_k`) — DCG/IDCG with binary gains")
      st.latex(r"\mathrm{nDCG@k}(q) = \frac{\mathrm{DCG@k}(q)}{\mathrm{IDCG@k}(q)}")

    root = _project_root()
    dataset_dir = (
        root
        / ".cache"
        / "data"
        / "piotr-rybak__poleval2022-passage-retrieval-dataset"
        / "wiki-trivia"
    )

    st.subheader("Dataset location (wiki-trivia only)")
    st.code(str(dataset_dir))


page()
