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
    st.title("Final Summary")
    st.caption("Best method is selected by ndcg_at_k, within comparable runs.")

    st.markdown(
        """
### How to read this summary (repo rationale)

- The repository treats **nDCG@10** as the primary decision metric for PolEval passage retrieval.
- Supporting metrics (Hits/Recall/Precision/MRR) are logged to help diagnose whether improvements come from
  “more hits anywhere in top‑10” vs “better ordering at the top”.

Sources (repo docs):
- `src/eval/README.md`
- `results/README.md`
"""
    )

    df = _wiki_trivia_only(_load_all_metrics())
    if df.empty:
        st.error("No metrics found for wiki-trivia in .cache/submissions/**/metrics.csv")
        return

    methods = [
        "tfidf_cosine",
        "bm25_okapi",
        "hybrid_bm25_biencoder",
        "hybrid_tfidf_biencoder",
    ]
    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        st.error("No eligible runs found for summary.")
        return

    df["ndcg_at_k"] = pd.to_numeric(df["ndcg_at_k"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    st.subheader("Comparable runs (wiki-trivia)")
    show_cols = [
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
        "biencoder_device",
        "biencoder_batch_size",
    ]
    st.dataframe(
        df[show_cols].sort_values(["k", "ndcg_at_k"], ascending=[True, False]),
        width="stretch",
    )

    st.subheader("Best method (by ndcg_at_k)")
    comparable = df.dropna(subset=["ndcg_at_k", "k"]).copy()
    if comparable.empty:
        st.info("Cannot select best: missing ndcg_at_k or k.")
        return

    best_rows = []
    for k_val, group in comparable.groupby("k", dropna=False):
        group = group.dropna(subset=["ndcg_at_k"])
        if group.empty:
            continue
        best_rows.append(group.sort_values("ndcg_at_k", ascending=False).iloc[0])

    best = pd.DataFrame(best_rows)
    st.dataframe(best[["k", "method", "ndcg_at_k"]], width="stretch")

    st.subheader("Trade-offs visible in logs")
    st.markdown(
        """
- **Lexical baselines** have fewer runtime knobs in logs (e.g., BM25 has `bm25_k1`, `bm25_b`).
- **Hybrid runs** introduce a 2-stage pipeline with additional parameters:
  - Stage 1 size: `top_k_candidates`
  - Reranking size: `rerank_k`
  - Bi-encoder execution: `biencoder_device`, `biencoder_batch_size`, `biencoder_max_length`

This page reports only what is directly present in `metrics.csv`.

Repo note on observed behavior (current cached runs):
- The `results/README.md` discusses a concrete pattern seen in these runs: hybrid reranking can improve
    `hits_at_k` while not necessarily improving rank-sensitive metrics (`mrr_at_k`, `ndcg_at_k`).

Sources (repo docs):
- `results/README.md`
"""
    )


page()
