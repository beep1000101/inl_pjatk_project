from __future__ import annotations

from pathlib import Path

import pandas as pd
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

This Streamlit presentation is **scoped strictly to the `wiki-trivia` subdataset** from the cached dataset snapshot:

- `.cache/data/piotr-rybak__poleval2022-passage-retrieval-dataset/wiki-trivia`

The repository codebase supports cross-domain evaluation, but this app **ignores** all non‑`wiki-trivia` data and results.

## Dataset context (wiki-trivia)

For `wiki-trivia`, the corpus is a large collection of **Wikipedia passages** (split into paragraph-like chunks).
The queries are **trivia / general-knowledge questions** (quiz-show style), and the retrieval system must return
the **top-10 passage IDs** that most likely contain the answer.

This context is why the repo emphasizes two families of approaches:
- **Lexical methods (TF‑IDF, BM25)** are fast, interpretable, and training-free, but they depend on word overlap
  and do not model word order.
- **Neural/semantic reranking** can be more “meaning aware”, but is expensive to run over the full corpus.
  That’s why the hybrid pipelines first use a lexical stage as a funnel, then rerank only a small candidate set.
"""
    )

    st.subheader("Toy example (questions + passages)")
    st.caption("Illustrative only (not real dataset rows)")

    st.markdown("**Example questions**")
    toy_questions = pd.DataFrame(
        [
            {"question_id": "q-001", "text": "What is the capital of France?"},
            {"question_id": "q-002", "text": "Which planet is known as the Red Planet?"},
            {"question_id": "q-003", "text": "Who wrote 'Pan Tadeusz'?"},
        ]
    )
    st.dataframe(toy_questions, width="stretch")

    st.markdown("**Example passages (Wikipedia-like)**")
    toy_passages = pd.DataFrame(
        [
            {
                "passage_id": "p-10",
                "title": "Paris",
                "text": "Paris is the capital and most populous city of France.",
            },
            {
                "passage_id": "p-22",
                "title": "Mars",
                "text": "Mars is often called the Red Planet because of its reddish appearance.",
            },
            {
                "passage_id": "p-31",
                "title": "Adam Mickiewicz",
                "text": "Adam Mickiewicz wrote the epic poem 'Pan Tadeusz'.",
            },
            {
                "passage_id": "p-40",
                "title": "France",
                "text": "France is a country in Western Europe with many cities and regions.",
            },
        ]
    )
    st.dataframe(toy_passages, width="stretch")

    st.markdown(
        """
**What the system outputs**

For each question, the system returns an ordered list of 10 passage IDs (top‑10). For example:

- for `q-001`: `[p-10, p-40, ...]` (10 total IDs)
"""
    )

    st.markdown(
        r"""
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

## Metrics used in this app (formulas, no set notation)

All metrics are computed **at cutoff $k$** (typically $k=10$) and then averaged.

Implementation details (as implemented in the repo):
- Relevance is **binary** from `pairs-*.tsv`. If a `score` column exists, only rows with `score > 0` count as relevant.
- Metrics are computed only for **labeled questions** and averaged over those questions.

Define:
- $N$: number of labeled questions.
- For each labeled question $q$, the model returns a ranked list of top-$k$ passage IDs $p_{q,1},\dots,p_{q,k}$.
- $y_{q,i}\in\{0,1\}$: whether the passage at rank $i$ is relevant for question $q$.
- $n_{rel}(q)$: the number of relevant passages for question $q$ in the labels.

The repository’s logged metrics correspond to:

- **Hits@k** (`hits_at_k`)
- **Recall@k** (`recall_at_k`)
- **Precision@k** (`precision_at_k`)
- **MRR@k** (`mrr_at_k`)
- **nDCG@k** (`ndcg_at_k`)

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

    st.subheader("Metric formulas")

    st.markdown("**Hits@k** (`hits_at_k`) — averaged over labeled questions")
    st.latex(
      r"\mathrm{Hits@k} = \frac{1}{N}\sum_{q=1}^{N} \mathbb{1}\left[\sum_{i=1}^{k} y_{q,i} > 0\right]"
    )

    st.markdown("**Recall@k** (`recall_at_k`) — averaged over labeled questions")
    st.latex(
      r"\mathrm{Recall@k} = \frac{1}{N}\sum_{q=1}^{N} \frac{\sum_{i=1}^{k} y_{q,i}}{\max(1,n_{rel}(q))}"
    )

    st.markdown("**Precision@k** (`precision_at_k`) — averaged over labeled questions")
    st.latex(
      r"\mathrm{Precision@k} = \frac{1}{N}\sum_{q=1}^{N} \frac{1}{k}\sum_{i=1}^{k} y_{q,i}"
    )

    st.markdown("**MRR@k** (`mrr_at_k`) — averaged over labeled questions")
    st.latex(
      r"\mathrm{MRR@k} = \frac{1}{N}\sum_{q=1}^{N} \left(\max_{1\le i\le k} \frac{y_{q,i}}{i}\right)"
    )

    st.markdown("**nDCG@k** (`ndcg_at_k`) — averaged over labeled questions")
    st.latex(r"\mathrm{DCG@k}(q) = \sum_{i=1}^{k} \frac{y_{q,i}}{\log_2(i+1)}")
    st.latex(r"\mathrm{IDCG@k}(q) = \sum_{i=1}^{\min(k,n_{rel}(q))} \frac{1}{\log_2(i+1)}")
    st.latex(
      r"\mathrm{nDCG@k} = \frac{1}{N}\sum_{q=1}^{N} \frac{\mathrm{DCG@k}(q)}{\mathrm{IDCG@k}(q)}"
    )

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
