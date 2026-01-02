from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ALLOWED_FIELDS = [
    "run_id",
    "method",
    "dataset_id",
    "subdataset",
    "questions_split",
    "pairs_split",
    "k",
    "n_questions",
    "n_labeled",
    "hits_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "out_tsv",
    "submission_only",
    "top_k_candidates",
    "chunk_size",
    "rerank_k",
    "alpha",
    "bm25_k1",
    "bm25_b",
    "biencoder_model",
    "biencoder_device",
    "biencoder_batch_size",
    "biencoder_max_length",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def _load_all_metrics() -> pd.DataFrame:
    root = _project_root()
    submissions_dir = root / ".cache" / "submissions"
    metric_files = sorted(submissions_dir.glob("**/metrics.csv"))

    frames: list[pd.DataFrame] = []
    for path in metric_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["_metrics_path"] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    for col in ALLOWED_FIELDS:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined = combined[[c for c in combined.columns if c in set(ALLOWED_FIELDS + ["_metrics_path"])]]
    return combined


def _wiki_trivia_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    return df[
        (df["dataset_id"] == "piotr-rybak__poleval2022-passage-retrieval-dataset")
        & (df["subdataset"] == "wiki-trivia")
    ].copy()


def page() -> None:
    st.title("Lexical + Semantic Re-ranking (Hybrid)")
    st.caption("Primary metric: ndcg_at_k (from metrics.csv)")

    st.markdown(
        """
### Why a 2-stage hybrid pipeline (repo rationale)

The repository’s hybrid systems are designed around a practical constraint: the `wiki-trivia` corpus has
millions of passages, so you cannot afford to run a semantic model over the full corpus per query.

Instead, the repo uses:

1) **Stage 1: lexical candidate generation** (BM25 or TF‑IDF) to retrieve a manageable shortlist
2) **Stage 2: semantic reranking** (bi-encoder) over only those candidates

This matches how the repo describes preprocessing/evaluation:
- caching is required to make experiments feasible,
- reranking is only meaningful if the relevant passage is present in the candidate set.

Sources (repo docs):
- `src/preprocess/README.md`
- `src/eval/README.md`
- `src/calibration/README.md`
"""
    )

    df = _wiki_trivia_only(_load_all_metrics())
    if df.empty:
        st.error("No metrics found for wiki-trivia in .cache/submissions/**/metrics.csv")
        return

    hybrid = df[df["method"].isin(["hybrid_bm25_biencoder", "hybrid_tfidf_biencoder"])].copy()
    if hybrid.empty:
        st.error("No hybrid runs found (hybrid_*_biencoder).")
        return

    st.subheader("Runs (wiki-trivia)")
    show_cols = [
        "run_id",
        "method",
        "dataset_id",
        "subdataset",
        "questions_split",
        "pairs_split",
        "k",
        "ndcg_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "hits_at_k",
        "top_k_candidates",
        "rerank_k",
        "chunk_size",
        "alpha",
        "bm25_k1",
        "bm25_b",
        "biencoder_model",
        "biencoder_device",
        "biencoder_batch_size",
        "biencoder_max_length",
        "submission_only",
    ]
    st.dataframe(hybrid[show_cols], width="stretch")

    st.subheader("Pipeline parameters")
    st.markdown(
        """
This repository implements a 2-stage hybrid pipeline:

- Stage 1 (lexical): retrieve `top_k_candidates` passages.
- Stage 2 (semantic): rerank the top `rerank_k` candidates with a bi-encoder.

The logged fields above also capture runtime-related knobs:
- `chunk_size` for corpus processing
- bi-encoder batching: `biencoder_batch_size`, `biencoder_max_length`
- execution device: `biencoder_device`

Note on `alpha`:
- In the currently cached hybrid runs, `alpha` is empty in `metrics.csv`. The repo’s `results/README.md`
    documents this as `alpha=None`, meaning ranking is driven by the bi-encoder score and lexical score acts
    as a tie-breaker.

Sources (repo docs):
- `results/README.md`
- `results/hybrid/hybrid_bm25_biencoder.md`
- `results/hybrid/hybrid_tfidf_biencoder.md`
"""
    )


page()
