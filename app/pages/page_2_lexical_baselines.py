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
    "submission_only",
    "k",
    "n_questions",
    "n_labeled",
    "hits_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
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


@st.cache_data(show_spinner=False)
def _load_calibration_hits_points() -> pd.DataFrame:
    root = _project_root()
    path = (
        root
        / ".cache"
        / "calibration"
        / "wiki-trivia"
        / "test"
        / "hits_points_maxk200_log_p20.csv"
    )
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["_calibration_path"] = str(path)
    return df


def page() -> None:
    st.title("Lexical Baselines")
    st.caption("Primary metric: ndcg_at_k (from metrics.csv)")

    st.markdown(
        """
### Why lexical baselines come first (repo rationale)

This repository treats TF‑IDF and BM25 as the foundational baselines because:

- They are **training-free** and therefore robust in **zero-shot** / cross-domain settings.
- They scale to a multi-million passage corpus once passage representations are **cached**.

This is explicitly part of the project’s design: separate the pipeline into stages, cache expensive artifacts,
and keep evaluation reproducible.

Sources (repo docs):
- `src/README.md`
- `src/preprocess/README.md`
- `src/eval/README.md`
"""
    )

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
        "dataset_id",
        "subdataset",
        "questions_split",
        "pairs_split",
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
        "submission_only",
        "chunk_size",
    ]
    st.dataframe(lexical[show_cols], width="stretch")

    st.subheader("Calibration: Hits@k vs k (wiki-trivia)")
    st.caption("Field: hits_at_k. Source: .cache/calibration/wiki-trivia/test/hits_points_maxk200_log_p20.csv")

    cal = _load_calibration_hits_points()
    if cal.empty:
        st.info(
            "No calibration file found at .cache/calibration/wiki-trivia/test/hits_points_maxk200_log_p20.csv"
        )
    else:
        cal = cal[cal["method"].isin(["bm25", "tfidf"])].copy()
        cal["k"] = pd.to_numeric(cal["k"], errors="coerce")
        cal["hits_at_k"] = pd.to_numeric(cal["hits_at_k"], errors="coerce")
        cal = cal.dropna(subset=["k", "hits_at_k"]).sort_values(["method", "k"])

        pivot = cal.pivot_table(index="k", columns="method", values="hits_at_k", aggfunc="mean")
        st.line_chart(pivot)

        with st.expander("Show calibration table"):
            # Only show columns that are in the allow-list.
            cal_show = [c for c in ["method", "k", "hits_at_k"] if c in cal.columns]
            st.dataframe(cal[cal_show], width="stretch")

        st.markdown(
            """
Why this calibration exists (repo rationale):

- The project uses calibration to make a data-driven choice of *how large* a retrieval prefix/candidate set
  needs to be before improvements taper off.
- This is framed as an engineering trade-off: larger $k$ improves Hits@k up to a point, but increases runtime
  and memory usage (especially for reranking pipelines).

Source (repo docs):
- `src/calibration/README.md`
"""
        )

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
    st.dataframe(best[["method", "k", "ndcg_at_k"]], width="stretch")


page()
