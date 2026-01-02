from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import CACHE_DIR, poleval2022_pairs_path, poleval2022_questions_path
from src.preprocess.tf_idf_vectors import read_jsonl


RetrieverFn = Callable[[list[str], int], np.ndarray]
ProgressFn = Callable[[int, int], None]

logger = logging.getLogger(__name__)


def _l2_normalize_rows(x: Any, *, eps: float = 1e-12) -> np.ndarray:
    """L2 normalize rows and return a writable float32 array.

    We intentionally use NumPy here instead of `faiss.normalize_L2` because
    `faiss.normalize_L2` can segfault on some platforms/inputs.

    Important: inputs can be read-only views (e.g., memmap slices), so we
    always copy to a writable float32 array.
    """

    arr = np.array(x, dtype=np.float32, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape}")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr /= np.maximum(norms, eps)
    return arr


@dataclass(frozen=True)
class EvalResult:
    out_path: Path
    k: int
    n_questions: int
    hits_at_k: float | None = None
    recall_at_k: float | None = None
    precision_at_k: float | None = None
    mrr_at_k: float | None = None
    ndcg_at_k: float | None = None
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


def compute_metrics_at_k(
    preds_df: pd.DataFrame,
    rel_by_q: dict[int, set[str]],
    *,
    k: int,
) -> tuple[
    float | None,  # hits@k
    float | None,  # recall@k
    float | None,  # precision@k
    float | None,  # mrr@k
    float | None,  # ndcg@k
    int,
]:
    """Compute common IR metrics at k using binary relevance.

    Definitions (averaged over labeled questions):
    - Hits@k: 1 if any relevant in top-k else 0
    - Recall@k: |relevant ∩ top-k| / |relevant|
    - Precision@k: |relevant ∩ top-k| / k
    - MRR@k: 1 / rank of first relevant (within top-k), else 0
    - nDCG@k: DCG/IDCG with binary gains
    """

    hit_sum = 0.0
    recall_sum = 0.0
    precision_sum = 0.0
    rr_sum = 0.0
    ndcg_sum = 0.0
    n = 0

    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float32))

    for qid, row in preds_df.iterrows():
        rel = rel_by_q.get(int(str(qid)))
        if not rel:
            continue

        n += 1
        ranked = [str(x) for x in row.tolist()[:k] if str(x)]

        hits = 0
        first_rank = None
        dcg = 0.0

        for i, pid in enumerate(ranked[:k], start=1):
            if pid in rel:
                hits += 1
                if first_rank is None:
                    first_rank = i
                dcg += float(discounts[i - 1])

        if hits > 0:
            hit_sum += 1.0
            rr_sum += 1.0 / \
                float(first_rank if first_rank is not None else k + 1)

        recall_sum += hits / max(1, len(rel))
        precision_sum += hits / k

        idcg_len = min(k, len(rel))
        if idcg_len > 0:
            idcg = float(discounts[:idcg_len].sum())
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

    if n == 0:
        return None, None, None, None, None, 0

    return (
        hit_sum / n,
        recall_sum / n,
        precision_sum / n,
        rr_sum / n,
        ndcg_sum / n,
        n,
    )


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
    run_name: str | None = None,
) -> EvalResult:
    logger.info(
        "Starting retrieval: subdataset=%s questions_split=%s k=%s",
        subdataset,
        questions_split,
        k,
    )
    questions_df = load_questions(dataset_id, subdataset, questions_split)
    qids = questions_df.index.to_numpy(dtype=int)
    q_texts = questions_df["text"].tolist()

    logger.info("Loaded %s questions", len(q_texts))

    last_log_t = time.monotonic()

    def progress(done: int, total: int) -> None:
        nonlocal last_log_t
        now = time.monotonic()
        if total <= 0:
            return
        # rate-limit logs; always show completion
        if now - last_log_t >= 2.0 or done >= total:
            pct = 100.0 * float(done) / float(total)
            logger.info("Retrieval progress: %s/%s (%.1f%%)", done, total, pct)
            last_log_t = now

    # Allow retrievers to optionally accept a progress callback.
    # Fallback cleanly for simple retrievers with signature (texts, k).
    try:
        # Some retrievers accept a progress callback; keep this optional.
        pred_ids = retriever(q_texts, k, progress=progress)  # type: ignore
    except TypeError:
        pred_ids = retriever(q_texts, k)

    pred_ids = _to_2d_str_array(pred_ids, n=len(q_texts), k=k)

    preds_df = pd.DataFrame(pred_ids, index=qids, columns=[
                            f"rank_{j}" for j in range(1, k + 1)])

    if out_path is None:
        prefix = run_name or "tfidf"
        out_dir = CACHE_DIR / "submissions" / prefix
        out_path = out_dir / f"{subdataset}_questions-{questions_split}.tsv"

    write_submission_tsv(preds_df, out_path)
    logger.info("Wrote TSV: %s", out_path)

    hits_at_k = None
    recall_at_k = None
    precision_at_k = None
    mrr_at_k = None
    ndcg_at_k = None
    n_labeled = None
    if pairs_split is not None:
        rel_by_q = load_relevance_pairs(dataset_id, subdataset, pairs_split)
        logger.info("Loaded relevance labels for %s questions", len(rel_by_q))
        (
            hits_at_k,
            recall_at_k,
            precision_at_k,
            mrr_at_k,
            ndcg_at_k,
            n_labeled,
        ) = compute_metrics_at_k(preds_df, rel_by_q, k=k)

        if hits_at_k is not None:
            logger.info(
                "Metrics@%s: Hits=%.4f Recall=%.4f Precision=%.4f MRR=%.4f nDCG=%.4f",
                k,
                hits_at_k,
                recall_at_k,
                precision_at_k,
                mrr_at_k,
                ndcg_at_k,
            )

    return EvalResult(
        out_path=out_path,
        k=k,
        n_questions=len(q_texts),
        hits_at_k=hits_at_k,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        mrr_at_k=mrr_at_k,
        ndcg_at_k=ndcg_at_k,
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
    progress: ProgressFn | None = None,
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

        if progress is not None:
            progress(end, N)

    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)

    return passage_ids[top_idx]


def embed_fasttext_avg(
    *,
    model: Any,
    texts: list[str],
    tokenize: Callable[[str], list[str]],
    dtype: Any = np.float32,
) -> np.ndarray:
    """Embeds texts as average of FastText word vectors.

    Returns shape: (len(texts), dim)
    """

    dim = int(model.get_dimension())
    out = np.zeros((len(texts), dim), dtype=dtype)

    for i, text in enumerate(texts):
        tokens = tokenize(str(text))
        if not tokens:
            continue

        v = np.zeros(dim, dtype=np.float32)
        for t in tokens:
            v += model.get_word_vector(t).astype(np.float32, copy=False)
        v /= float(len(tokens))
        out[i] = v.astype(dtype, copy=False)

    return out


def retrieve_dense_cosine_topk(
    *,
    passage_vectors: np.ndarray,
    passage_ids: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 10,
    chunk_size: int = 50_000,
    eps: float = 1e-12,
    progress: ProgressFn | None = None,
) -> np.ndarray:
    """Chunked cosine top-k for dense vectors.

    passage_vectors: (N, D)
    query_vectors:   (nq, D)
    Returns: (nq, k) passage id strings
    """

    X = passage_vectors
    Q = query_vectors.astype(np.float32, copy=False)
    nq = Q.shape[0]
    N = X.shape[0]

    # normalize queries
    q_norm = np.linalg.norm(Q, axis=1, keepdims=True)
    Qn = Q / np.maximum(q_norm, eps)

    top_scores = np.full((nq, k), -np.inf, dtype=np.float32)
    top_idx = np.full((nq, k), -1, dtype=np.int32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Xb = np.asarray(X[start:end], dtype=np.float32)
        if Xb.size == 0:
            continue

        x_norm = np.linalg.norm(Xb, axis=1, keepdims=True)
        Xbn = Xb / np.maximum(x_norm, eps)

        sims = Xbn @ Qn.T  # (B, nq)
        B = sims.shape[0]
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

        if progress is not None:
            progress(end, N)

    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)

    return passage_ids[top_idx]


def build_faiss_flat_ip_index(
    *,
    passage_vectors: np.ndarray,
    chunk_size: int = 200_000,
    progress: ProgressFn | None = None,
) -> Any:
    """Build a FAISS IndexFlatIP for cosine search.

    We L2-normalize vectors before adding them, so inner product == cosine.

    Note: IndexFlatIP stores all vectors in RAM (can be large for 6.6M x 300).
    """

    try:
        import faiss  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "faiss is required for FAISS retrieval. Install faiss-cpu."
        ) from e

    X = passage_vectors
    if X.ndim != 2:
        raise ValueError(f"passage_vectors must be 2D, got shape={X.shape}")

    N, dim = int(X.shape[0]), int(X.shape[1])
    index = faiss.IndexFlatIP(dim)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Xb = np.asarray(X[start:end], dtype=np.float32)
        if Xb.size == 0:
            continue

        Xbn = _l2_normalize_rows(Xb)
        index.add(Xbn)  # type: ignore

        if progress is not None:
            progress(end, N)

    return index


def build_faiss_ivfpq_ip_index(
    *,
    passage_vectors: np.ndarray,
    nlist: int = 4096,
    m: int = 30,
    nbits: int = 8,
    train_size: int = 200_000,
    chunk_size: int = 200_000,
    seed: int = 0,
    progress: ProgressFn | None = None,
) -> Any:
    """Build a FAISS IVF-PQ index for cosine search (via inner product).

    Compared to IndexFlatIP, this is much smaller in RAM and usually avoids
    kernel OOM for multi-million passage collections.

    Notes:
    - We L2-normalize vectors before training/adding, so inner product == cosine.
    - Requires training on a sample of vectors.
    - Tune recall/speed via `index.nprobe` (e.g. 16..128).
    """

    try:
        import faiss  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "faiss is required for FAISS retrieval. Install faiss-cpu."
        ) from e

    X = passage_vectors
    if X.ndim != 2:
        raise ValueError(f"passage_vectors must be 2D, got shape={X.shape}")

    N, dim = int(X.shape[0]), int(X.shape[1])
    if dim % m != 0:
        raise ValueError(f"dim={dim} must be divisible by m={m} for PQ")

    rng = np.random.default_rng(seed)
    train_n = min(int(train_size), N)
    if train_n <= 0:
        raise ValueError("train_size must be > 0")

    # Sample without replacement (training subset)
    train_idx = rng.choice(N, size=train_n, replace=False)
    Xt = _l2_normalize_rows(np.asarray(X[train_idx], dtype=np.float32))

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        int(nlist),
        int(m),
        int(nbits),
        faiss.METRIC_INNER_PRODUCT,
    )

    logger.info(
        "Training FAISS IVF-PQ: N=%s dim=%s nlist=%s m=%s nbits=%s train_n=%s",
        N,
        dim,
        nlist,
        m,
        nbits,
        train_n,
    )
    index.train(Xt)  # type: ignore

    logger.info("Adding vectors to FAISS index...")
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        Xb = np.asarray(X[start:end], dtype=np.float32)
        if Xb.size == 0:
            continue

        Xbn = _l2_normalize_rows(Xb)
        index.add(Xbn)  # type: ignore

        if progress is not None:
            progress(end, N)

    return index


def retrieve_dense_faiss_topk(
    *,
    index: Any,
    passage_ids: np.ndarray,
    query_vectors: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Search a FAISS index and return (nq, k) passage ids.

    Assumes the index was built over L2-normalized passage vectors.
    We L2-normalize queries here to do cosine similarity via inner product.
    """

    try:
        import faiss  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "faiss is required for FAISS retrieval. Install faiss-cpu."
        ) from e

    Q = np.asarray(query_vectors, dtype=np.float32)
    if Q.ndim != 2:
        raise ValueError(f"query_vectors must be 2D, got shape={Q.shape}")

    Qn = _l2_normalize_rows(Q)
    _, idx = index.search(Qn, k)  # type: ignore
    idx = np.asarray(idx)

    # Map indices to ids; handle any -1 (faiss returns -1 if not enough vectors)
    out = np.empty(idx.shape, dtype=passage_ids.dtype)
    out[:] = ""
    mask = idx >= 0
    out[mask] = passage_ids[idx[mask]]
    return out
