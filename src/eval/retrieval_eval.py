from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import CACHE_DIR, poleval2022_pairs_path, poleval2022_questions_path
from src.preprocess.tf_idf_vectors import read_jsonl


RetrieverFn = Callable[[list[str], int], np.ndarray]


@dataclass(frozen=True)
class EvalResult:
    out_path: Path
    k: int
    n_questions: int
    hits_at_k: float | None = None
    mrr_at_k: float | None = None
    n_labeled: int | None = None


def load_questions(dataset_id: str, subdataset: str, split: str) -> pd.DataFrame:
    questions_path = poleval2022_questions_path(dataset_id, subdataset, split)
    df = read_jsonl(questions_path).set_index("id")
    df.index = df.index.astype(int)
    df["text"] = df["text"].astype(str)
    return df


def load_relevance_pairs(dataset_id: str, subdataset: str, split: str) -> dict[int, set[str]]:
    pairs_path = poleval2022_pairs_path(dataset_id, subdataset, split=split)
    if not pairs_path.exists():
        return {}

    pairs_df = pd.read_csv(pairs_path, sep="\t")
    pairs_df = pairs_df.rename(
        columns={"question-id": "question_id", "passage-id": "passage_id"})

    if "score" in pairs_df.columns:
        pairs_df["score"] = pd.to_numeric(pairs_df["score"], errors="coerce")
        pairs_df = pairs_df[pairs_df["score"].fillna(0) > 0]

    rel_by_q = (
        pairs_df.groupby("question_id")["passage_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    return {int(qid): rel for qid, rel in rel_by_q.items()}


def _to_2d_str_array(preds: Any, *, n: int, k: int) -> np.ndarray:
    arr = np.asarray(preds)

    if arr.ndim == 1:
        # allow list[list[str]] encoded strangely
        arr = np.asarray(list(preds), dtype=object)

    if arr.ndim != 2:
        raise ValueError(
            f"preds must be 2D array-like, got shape={getattr(arr, 'shape', None)}")

    if arr.shape[0] != n:
        raise ValueError(
            f"preds first dim must be n_questions={n}, got {arr.shape[0]}")

    if arr.shape[1] < k:
        pad = np.full((n, k - arr.shape[1]), "", dtype=object)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > k:
        arr = arr[:, :k]

    return arr.astype(str, copy=False)


def compute_hits_mrr_at_k(
    preds_df: pd.DataFrame,
    rel_by_q: dict[int, set[str]],
    *,
    k: int,
) -> tuple[float | None, float | None, int]:
    hits = 0
    rr_sum = 0.0
    n = 0

    for qid, row in preds_df.iterrows():
        qid_int = int(str(qid))  # pandas index can be typed as Hashable
        rel = rel_by_q.get(qid_int)
        if not rel:
            continue

        n += 1
        ranked = [str(x) for x in row.tolist()[:k] if str(x)]
        hit_rank = None
        for r, pid in enumerate(ranked, start=1):
            if pid in rel:
                hit_rank = r
                break

        if hit_rank is not None:
            hits += 1
            rr_sum += 1.0 / hit_rank

    if n == 0:
        return None, None, 0

    return hits / n, rr_sum / n, n


def write_submission_tsv(preds_df: pd.DataFrame, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_path, sep="\t", header=False, index=False)
    return out_path


def evaluate_and_write_submission(
    *,
    dataset_id: str,
    subdataset: str,
    questions_split: str,
    retriever: RetrieverFn,
    k: int = 10,
    pairs_split: str | None = None,
    out_path: Path | None = None,
) -> EvalResult:
    questions_df = load_questions(dataset_id, subdataset, questions_split)
    qids = questions_df.index.to_numpy(dtype=int)
    q_texts = questions_df["text"].tolist()

    pred_ids = retriever(q_texts, k)
    pred_ids = _to_2d_str_array(pred_ids, n=len(q_texts), k=k)

    preds_df = pd.DataFrame(pred_ids, index=qids, columns=[
                            f"rank_{j}" for j in range(1, k + 1)])

    if out_path is None:
        out_dir = CACHE_DIR / "submissions"
        out_path = out_dir / \
            f"tfidf_{subdataset}_questions-{questions_split}.tsv"

    write_submission_tsv(preds_df, out_path)

    hits_at_k = None
    mrr_at_k = None
    n_labeled = None
    if pairs_split is not None:
        rel_by_q = load_relevance_pairs(dataset_id, subdataset, pairs_split)
        hits_at_k, mrr_at_k, n_labeled = compute_hits_mrr_at_k(
            preds_df, rel_by_q, k=k)

    return EvalResult(
        out_path=out_path,
        k=k,
        n_questions=len(q_texts),
        hits_at_k=hits_at_k,
        mrr_at_k=mrr_at_k,
        n_labeled=n_labeled,
    )


def retrieve_tfidf_topk(
    *,
    vectorizer: Any,
    passages_matrix: Any,
    passage_ids: np.ndarray,
    query_texts: list[str],
    k: int = 10,
    chunk_size: int = 10_000,
) -> np.ndarray:
    """Vectorized TF-IDF retrieval.

    Returns: (n_queries, k) passage id strings.

    Notes:
    - Computes sims per chunk as dense (chunk_size x n_queries); tune chunk_size for RAM.
    """

    Q = vectorizer.transform(query_texts)  # (nq, D) sparse
    nq = Q.shape[0]
    X = passages_matrix  # (N, D) sparse
    N = X.shape[0]

    top_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    top_idx = np.full((nq, k), -1, dtype=np.int32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Xb = X[start:end]
        sims = (Xb @ Q.T).toarray().astype(np.float32, copy=False)  # (B, nq)
        B = sims.shape[0]
        if B == 0:
            continue

        k_local = min(k, B)
        idx_local = np.argpartition(-sims, kth=k_local -
                                    1, axis=0)[:k_local, :]
        scores_local = np.take_along_axis(sims, idx_local, axis=0)
        idx_local = idx_local + start

        comb_scores = np.concatenate([top_scores.T, scores_local], axis=0)
        comb_idx = np.concatenate([top_idx.T, idx_local], axis=0)
        sel = np.argpartition(-comb_scores, kth=k - 1, axis=0)[:k, :]

        top_scores = np.take_along_axis(comb_scores, sel, axis=0).T
        top_idx = np.take_along_axis(comb_idx, sel, axis=0).T

    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)

    return passage_ids[top_idx]
