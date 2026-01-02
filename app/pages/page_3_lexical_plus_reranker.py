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
def _load_toy_passages() -> pd.DataFrame:
    root = _project_root()
    path = root / "app" / "data" / "toy_passages.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
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
        "n_questions",
        "n_labeled",
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

    st.subheader("Bi-encoder used (and why)")
    model_names = sorted(
        {
            str(x)
            for x in hybrid.get("biencoder_model", pd.Series(dtype=object)).dropna().unique().tolist()
            if str(x).strip()
        }
    )
    if model_names:
        if len(model_names) == 1:
            st.markdown(f"Model in these logged runs: `{model_names[0]}`")
        else:
            st.markdown("Models in these logged runs:")
            st.markdown("\n".join([f"- `{m}`" for m in model_names]))
    else:
        st.markdown("Model name not present in the cached `metrics.csv` for these runs.")

    st.markdown(
        """
A **bi-encoder** encodes the query and each candidate passage *independently* into the same embedding space.
Then we score candidates by a simple similarity function (commonly cosine similarity / dot product).

Why it’s a good fit here (simple intuition):
- **Fast enough for reranking**: once lexical retrieval narrows the search to a few hundred candidates,
  a bi-encoder can score that shortlist efficiently in batches.
- **More expressive than pure lexical overlap**: it can still match paraphrases/semantic similarity when
  the exact keywords don’t align.

This repo also logs the practical knobs that keep this stage manageable (`biencoder_device`,
`biencoder_batch_size`, `biencoder_max_length`).
"""
    )

    st.subheader("Toy example: 2-stage hybrid retrieval")
    st.caption("Dummy passages → lexical top-j → bi-encoder rerank → top-k (indices used for lookup)")

    toy = _load_toy_passages()
    if toy.empty:
        st.info("Toy passages not found at app/data/toy_passages.csv")
        return

    st.markdown("**Dummy passage table (10 rows)**")
    st.dataframe(toy, width="stretch")

    query = st.text_input("Query", value="What is the capital of France?", key="toy_hybrid_query")

    j = 5
    top_k = 3

    st.markdown("↓ Stage 1: lexical retrieval (top-j candidates)")
    st.markdown(
        r"""
Represent the full corpus of 10 passages with a lexical vector-space matrix:

$$
M^{(lex)} \in \mathbb{R}^{10\times d_{lex}},\quad
M^{(lex)}=\begin{bmatrix}
m^{(lex)}_{0,0} & \cdots & m^{(lex)}_{0,d_{lex}-1}\\
\vdots & \ddots & \vdots\\
m^{(lex)}_{9,0} & \cdots & m^{(lex)}_{9,d_{lex}-1}
\end{bmatrix}
$$

and the query as a vector:

$$
v^{(lex)} \in \mathbb{R}^{d_{lex}},\quad
v^{(lex)}=[v^{(lex)}_0,\ldots,v^{(lex)}_{d_{lex}-1}]^\top
$$

Compute lexical cosine similarities:

$$
u^{(lex)} \in \mathbb{R}^{10},\quad
u^{(lex)}_i = \cos(M^{(lex)}_i, v^{(lex)}) = \frac{M^{(lex)}_i \cdot v^{(lex)}}{\lVert M^{(lex)}_i \rVert\,\lVert v^{(lex)} \rVert}
$$

(where $M^{(lex)}_i$ is the $i$-th row of $M^{(lex)}$).

Select the best $j$ candidates:

$$
\mathrm{indices}^{(lex)} = \operatorname{argmax}_{j}(u^{(lex)})
$$
"""
    )

    # Example candidate indices for demonstration.
    cand_indices = [2, 3, 8, 0, 9][:j]
    st.markdown(f"Example: **indices** (j={j}) = {cand_indices}")

    candidates = toy.iloc[cand_indices].copy()
    candidates.insert(0, "index", cand_indices)
    candidates.insert(1, "query", query)
    st.dataframe(candidates[["index", "query", "passage_id", "text"]], width="stretch")

    st.markdown("↓ Stage 2: bi-encoder reranking (top-k within candidates)")
    st.markdown(
        r"""
Now embed only the $j$ candidate passages with a bi-encoder (semantic space):

$$
M^{(sem)} \in \mathbb{R}^{j\times d_{sem}},\quad
M^{(sem)}=\begin{bmatrix}
m^{(sem)}_{0,0} & \cdots & m^{(sem)}_{0,d_{sem}-1}\\
\vdots & \ddots & \vdots\\
m^{(sem)}_{j-1,0} & \cdots & m^{(sem)}_{j-1,d_{sem}-1}
\end{bmatrix}
$$

and the query as a vector:

$$
v^{(sem)} \in \mathbb{R}^{d_{sem}},\quad
v^{(sem)}=[v^{(sem)}_0,\ldots,v^{(sem)}_{d_{sem}-1}]^\top
$$

Compute semantic cosine similarities over candidates:

$$
u^{(sem)} \in \mathbb{R}^{j},\quad
u^{(sem)}_r = \cos(M^{(sem)}_r, v^{(sem)}) = \frac{M^{(sem)}_r \cdot v^{(sem)}}{\lVert M^{(sem)}_r \rVert\,\lVert v^{(sem)} \rVert}
$$

(where $M^{(sem)}_r$ is the $r$-th row of $M^{(sem)}$, i.e., the $r$-th candidate in the stage-1 list).

Pick the best $k$ candidate rows and map them back to original passage indices:

$$
\mathrm{indices}^{(sem)} = \operatorname{argmax}_{k}(u^{(sem)}),\quad
\mathrm{indices}^{(top\text{-}k)} = \mathrm{indices}^{(lex)}[\mathrm{indices}^{(sem)}]
$$
"""
    )

    # Example rerank rows within the candidate list.
    rerank_rows = [0, 2, 1][:top_k]
    topk_indices = [cand_indices[r] for r in rerank_rows]
    st.markdown(f"Example: **indices** (k={top_k}) = {topk_indices}")

    topk = toy.iloc[topk_indices].copy()
    topk.insert(0, "index", topk_indices)
    topk.insert(1, "query", query)
    st.dataframe(topk[["index", "query", "passage_id", "text"]], width="stretch")


page()
