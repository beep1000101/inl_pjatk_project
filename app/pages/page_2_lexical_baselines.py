from __future__ import annotations

from pathlib import Path

import altair as alt
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
def _load_toy_passages() -> pd.DataFrame:
    root = _project_root()
    path = root / "app" / "data" / "toy_passages.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Keep a stable integer index for the "lookup indices" explanation.
    return df.reset_index(drop=True)


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

    st.subheader("How lexical spaces are created")
    st.markdown(
        r"""
This project precomputes lexical artifacts once and caches them under
`.cache/preprocessed_data/<method>/<subdataset>/...`.
That makes evaluation fast (no re-vectorizing millions of passages each run) and keeps a stable
row ordering so “top indices → lookup” is reproducible.

Common passage preprocessing (both TF‑IDF and BM25, from `src/preprocess/*`):
- Drop redirect-like entries where `text` starts with `REDIRECT` or `PATRZ`.
- Build the text used for vectorization as `title + " " + text`.
"""
    )

    st.markdown(
    r"""
#### TF‑IDF (used by `tfidf_cosine`)

Implementation (see `src/preprocess/tf_idf_vectors.py`):
- Uses `sklearn.feature_extraction.text.TfidfVectorizer` (defaults: `min_df=5`, `max_df=0.9`,
    `max_features=500_000`, `dtype=float32`).
- Fits on the passage corpus and produces a sparse TF‑IDF matrix $M \in \mathbb{R}^{N\times d}$.

How it works (in words):
- **TF** (term frequency) captures “does this passage talk about this word a lot?”.
- **IDF** (inverse document frequency) downweights words that appear everywhere (like stop-ish words)
    and upweights words that are rarer and therefore more discriminative.
- The result is a vector space where a query and passage are “close” if they share informative words.

Why it works for this project:
- PolEval questions often have strong lexical cues (names, entities, key phrases).
- TF‑IDF is training-free, quick to compute once cached, and a strong baseline for cross-domain retrieval.

Cached artifacts:
- `vectorizer.joblib` (fitted vectorizer)
- `passages_tfidf.npz` (sparse matrix)
- `passage_ids.npy` (row index → passage id)
- `meta.json` (parameters + shapes)
"""
    )

    st.markdown(
    r"""
#### BM25 (used by `bm25_okapi`)

Implementation (see `src/preprocess/bm25_vectors.py`):
- Builds a sparse **term-frequency** matrix using `sklearn.feature_extraction.text.CountVectorizer`
    with `token_pattern=r"(?u)\b\w+\b"`, `lowercase=True`, `dtype=int32`.
- Computes document lengths `doc_len` and an Okapi-style idf vector (saved to disk).

How it works (in words):
- BM25 is like TF‑IDF, but with two important tweaks:
    - **Saturation**: repeating a word many times helps, but with diminishing returns.
    - **Length normalization**: very long passages shouldn’t win just because they contain more words.

Why it works for this project:
- Wikipedia-like passages vary a lot in length; BM25’s normalization is a good fit.
- It’s still purely lexical (fast, training-free), so it’s a reliable first-stage retriever to generate
    candidates for rerankers.

Cached artifacts:
- `vectorizer.joblib`, `passages_tf.npz`, `passage_ids.npy`
- `doc_len.npy`, `idf.npy`, `meta.json`

Sources (repo docs/code):
- `src/preprocess/README.md`
- `src/preprocess/tf_idf_vectors.py`
- `src/preprocess/bm25_vectors.py`
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

    st.subheader("Toy example: lexical retrieval in vector space")
    st.caption("A minimal walkthrough: passages → (made-up) matrix M → query vector v → cosine similarity → top indices → lookup")

    toy = _load_toy_passages()
    if toy.empty:
        st.info("Toy passages not found at app/data/toy_passages.csv")
    else:
        st.markdown("**Dummy passage table (10 rows)**")
        st.dataframe(toy, width="stretch")

        query = st.text_input("Query", value="What is the capital of France?")

        st.markdown("↓ Cast passages to vector space")

        st.markdown(
            r"""
We represent the 10 passages as a matrix in some vector space:

$$
M \in \mathbb{R}^{10\times d},\quad
M=\begin{bmatrix}
m_{0,0} & \cdots & m_{0,d-1}\\
\vdots & \ddots & \vdots\\
m_{9,0} & \cdots & m_{9,d-1}
\end{bmatrix}
$$

and the query as a vector:

$$
v \in \mathbb{R}^{d},\quad
v = [v_0,\ldots,v_{d-1}]^\top
$$
"""
        )

        st.markdown("↓ Compute cosine similarity")
        st.markdown(
            r"""
Compute a score vector $u$ by taking cosine similarity between each row $M_i$ and the query vector $v$:

$$
u \in \mathbb{R}^{10},\quad
u_i = \cos(M_i, v) = \frac{M_i \cdot v}{\lVert M_i \rVert\,\lVert v \rVert}
$$

(where $M_i$ is the $i$-th row of $M$).
"""
        )

        st.markdown("↓ Pick best k indices")
        st.markdown(
            r"""
Pick the indices of the best $k$ scores:

$$
\mathrm{indices} = \operatorname{argmax}_k(u)
$$
"""
        )

        st.markdown("↓ Use these indices for lookup")
        top_k = 5
        # Example indices for demonstration (derived from u in a real system).
        indices = [2, 3, 8, 0, 9][:top_k]
        st.markdown(f"Example: **indices** = {indices}")

        lookup = toy.iloc[indices].copy()
        lookup.insert(0, "index", indices)
        lookup.insert(1, "query", query)
        st.dataframe(lookup[["index", "query", "passage_id", "text"]], width="stretch")

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

        # Plot metric (y) vs k (x). The calibration CSV only contains Hits@k curves,
        # so additional options are derived from the logged hits_at_k series.
        cal["hits_at_max_k"] = cal.groupby("method")["hits_at_k"].transform("max")
        cal["hits_norm_to_max_k"] = cal["hits_at_k"] / cal["hits_at_max_k"]
        cal["k_fraction_of_max"] = cal["k"] / cal["max_k"]
        cal["hits_per_k"] = cal["hits_at_k"] / cal["k"]

        cal = cal.sort_values(["method", "k"])
        cal["marginal_hits_gain"] = cal.groupby("method")["hits_at_k"].diff()
        cal["marginal_k"] = cal.groupby("method")["k"].diff()
        cal["marginal_hits_per_doc"] = cal["marginal_hits_gain"] / cal["marginal_k"]

        metric_label_to_col = {
            "hits_at_k": "hits_at_k",
            "hits_norm_to_max_k": "hits_norm_to_max_k",
            "hits_per_k": "hits_per_k",
            "marginal_hits_gain": "marginal_hits_gain",
            "marginal_hits_per_doc": "marginal_hits_per_doc",
            "k_fraction_of_max": "k_fraction_of_max",
        }
        selected_metric_label = st.selectbox(
            "Metric (y-axis)",
            options=list(metric_label_to_col.keys()),
            index=0,
        )
        metric_col = metric_label_to_col[selected_metric_label]

        plot_df = cal[["method", "k", metric_col]].dropna(subset=[metric_col]).copy()

        legend_toggle = alt.selection_point(fields=["method"], bind="legend")
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="k"),
                y=alt.Y(f"{metric_col}:Q", title=selected_metric_label),
                color=alt.Color(
                    "method:N",
                    title="method",
                    # User-requested palette: orange + white
                    scale=alt.Scale(domain=["bm25", "tfidf"], range=["#FF7F0E", "#FFFFFF"]),
                ),
                tooltip=[
                    alt.Tooltip("method:N"),
                    alt.Tooltip("k:Q"),
                    alt.Tooltip(f"{metric_col}:Q", format=".6f"),
                ],
            )
            .add_params(legend_toggle)
            .transform_filter(legend_toggle)
            .properties(height=320)
            .configure(background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="rgba(255,255,255,0.10)",
                tickColor="rgba(255,255,255,0.25)",
                domainColor="rgba(255,255,255,0.25)",
            )
            .configure_legend(
                labelColor="white",
                titleColor="white",
            )
        )
        st.altair_chart(chart, width="stretch")

        with st.expander("What do these calibration metrics mean?"):
            if hasattr(st, "page_link"):
                st.page_link(
                    "pages/page_1_problem_summary.py",
                    label="Open: Problem summary (metric formulas)",
                )
            else:
                st.markdown("See the **Problem summary** page for full metric formulas.")

            st.markdown(
                """
This calibration file contains a **Hits@k curve** (per method) and we plot a selected metric against $k$:

- **hits_at_k**: same as Hits@k from the Problem Summary page, evaluated for varying $k$.
- **hits_norm_to_max_k**: $\mathrm{Hits@k} / \max_{k'} \mathrm{Hits@k'}$ within the same method (normalizes to 1.0 at the best observed $k$).
- **hits_per_k**: $\mathrm{Hits@k} / k$ (a simple “hits per retrieved doc” proxy).
- **marginal_hits_gain**: change in Hits@k between two consecutive sampled $k$ points.
- **marginal_hits_per_doc**: $\Delta\mathrm{Hits@k} / \Delta k$ between sampled points (slope of the curve).
- **k_fraction_of_max**: $k / \mathrm{max\_k}$ (how far along the calibration range you are).

All derived metrics are computed **only from this calibration CSV**; the only “true” IR metric logged here is Hits@k.
"""
            )

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

    st.subheader("Best lexical baseline (by hits_at_k)")
    comparable = lexical.dropna(subset=["hits_at_k", "k"]).copy()
    if comparable.empty:
        st.info("Cannot determine best run: hits_at_k missing.")
        return

    # Ensure numeric compare
    comparable["hits_at_k"] = pd.to_numeric(comparable["hits_at_k"], errors="coerce")
    comparable["k"] = pd.to_numeric(comparable["k"], errors="coerce")

    # The repo logs show k=10 for both; keep grouping to avoid mismatched k comparisons.
    best_rows: list[pd.Series] = []
    for k_val, group in comparable.groupby("k", dropna=False):
        group = group.dropna(subset=["hits_at_k"])
        if group.empty:
            continue
        best_rows.append(group.sort_values("hits_at_k", ascending=False).iloc[0])

    if not best_rows:
        st.info("No comparable groups found to select a best run.")
        return

    best = pd.DataFrame(best_rows)
    st.dataframe(best[["method", "k", "hits_at_k"]], width="stretch")


page()
