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
    st.title("Lexical Baselines")
    st.caption("Primary metric: ndcg_at_k (from metrics.csv)")

    df = _wiki_trivia_only(_load_all_metrics())
    if df.empty:
        st.error("No metrics found for wiki-trivia in .cache/submissions/**/metrics.csv")
        return

    lexical = df[df["method"].isin(["tfidf_cosine", "bm25_okapi"])].copy()
    if lexical.empty:
        st.error("No lexical baseline runs found (tfidf_cosine / bm25_okapi).")
        return

    st.subheader("Runs (wiki-trivia)")
    show_cols = [
        "run_id",
        "method",
        "k",
        "n_questions",
        "n_labeled",
        "ndcg_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "hits_at_k",
        "bm25_k1",
        "bm25_b",
        "out_tsv",
        "submission_only",
        "chunk_size",
        "_metrics_path",
    ]
    st.dataframe(lexical[show_cols], use_container_width=True)

    st.subheader("Best lexical baseline (by ndcg_at_k)")
    comparable = lexical.dropna(subset=["ndcg_at_k", "k"]).copy()
    if comparable.empty:
        st.info("Cannot determine best run: ndcg_at_k missing.")
        return

    # Ensure numeric compare
    comparable["ndcg_at_k"] = pd.to_numeric(comparable["ndcg_at_k"], errors="coerce")
    comparable["k"] = pd.to_numeric(comparable["k"], errors="coerce")

    # The repo logs show k=10 for both; keep grouping to avoid mismatched k comparisons.
    best_rows: list[pd.Series] = []
    for k_val, group in comparable.groupby("k", dropna=False):
        group = group.dropna(subset=["ndcg_at_k"])
        if group.empty:
            continue
        best_rows.append(group.sort_values("ndcg_at_k", ascending=False).iloc[0])

    if not best_rows:
        st.info("No comparable groups found to select a best run.")
        return

    best = pd.DataFrame(best_rows)
    st.dataframe(best[["method", "k", "ndcg_at_k", "_metrics_path"]], use_container_width=True)

    st.markdown(
        """
### Calibration / tuning

- The repo contains a calibration module (`src/calibration/`), but the currently logged lexical runs in `.cache/submissions/` appear as single rows per method.
- If additional `metrics.csv` rows exist with different `bm25_k1` / `bm25_b` / `k`, they will appear automatically in the table above.
"""
    )


page()
