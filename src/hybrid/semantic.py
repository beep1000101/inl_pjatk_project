from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.eval.retrieval_eval import embed_fasttext_avg


@dataclass(frozen=True)
class RerankResult:
    indices: np.ndarray  # (nq, top_n)
    ids: np.ndarray  # (nq, top_n) dtype=str
    scores: np.ndarray  # (nq, top_n) float32


def _cosine_scores(
    *,
    query_vec: np.ndarray,  # (D,)
    cand_vecs: np.ndarray,  # (M, D)
    eps: float = 1e-12,
) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    x = np.asarray(cand_vecs, dtype=np.float32)

    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), eps)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return (xn @ qn.T).ravel().astype(np.float32, copy=False)


def _minmax01(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    denom = max(hi - lo, float(eps))
    return (x - lo) / denom


class FastTextCosineReranker:
    """Brute-force cosine reranker using fastText query embedding.

    Passage vectors are expected to be precomputed (see `src.preprocess.fasttext_vectors`).
    """

    def __init__(self, *, model: Any, tokenize: Any) -> None:
        self.model = model
        self.tokenize = tokenize

    def rerank(
        self,
        *,
        query_texts: list[str],
        candidate_indices: np.ndarray,  # (nq, K)
        candidate_lexical_scores: np.ndarray,  # (nq, K)
        passage_vectors: np.ndarray,  # (N, D)
        passage_ids: np.ndarray,  # (N,)
        top_n: int = 10,
        rerank_k: int | None = None,
        alpha: float | None = None,
    ) -> RerankResult:
        top_n = int(top_n)
        if top_n <= 0:
            raise ValueError("top_n must be > 0")

        nq, K = candidate_indices.shape
        out_idx = np.empty((nq, top_n), dtype=np.int32)
        out_scores = np.empty((nq, top_n), dtype=np.float32)
        # IMPORTANT: avoid `dtype=str` which creates a fixed-width unicode array
        # that may truncate ids (often to 1 char). Preserve full passage id strings.
        out_ids = np.empty((nq, top_n), dtype=np.asarray(passage_ids).dtype)

        q_vecs = embed_fasttext_avg(
            model=self.model,
            texts=[str(t) for t in query_texts],
            tokenize=self.tokenize,
            dtype=np.float32,
        )

        for i in range(nq):
            idx_row_full = np.asarray(candidate_indices[i], dtype=np.int32)
            lex_row_full = np.asarray(
                candidate_lexical_scores[i], dtype=np.float32)

            if rerank_k is not None:
                rk = int(rerank_k)
                if rk <= 0:
                    raise ValueError("rerank_k must be > 0")
                idx_row = idx_row_full[:rk]
                lex_row = lex_row_full[:rk]
            else:
                idx_row = idx_row_full
                lex_row = lex_row_full

            cand_vecs = np.asarray(passage_vectors[idx_row], dtype=np.float32)
            sem_scores = _cosine_scores(
                query_vec=q_vecs[i], cand_vecs=cand_vecs)
            pids = np.asarray(passage_ids)[idx_row].astype(str, copy=False)

            if alpha is None:
                # Deterministic ordering: semantic desc, lexical desc, passage_id asc
                order = np.lexsort((pids, -lex_row, -sem_scores))
            else:
                a = float(alpha)
                if not (0.0 <= a <= 1.0):
                    raise ValueError("alpha must be in [0, 1]")
                # Normalize scores per-query to a comparable range before mixing.
                lex_n = _minmax01(lex_row)
                sem_n = _minmax01(sem_scores)
                fused = (a * lex_n + (1.0 - a) *
                         sem_n).astype(np.float32, copy=False)
                # Deterministic ordering: fused desc, then semantic desc, then passage_id asc
                order = np.lexsort((pids, -sem_scores, -fused))

            order = order[:top_n]

            out_idx[i] = idx_row[order]
            # Preserve semantic scores as the returned score for interpretability.
            out_scores[i] = sem_scores[order]
            out_ids[i] = pids[order]

        return RerankResult(indices=out_idx, ids=out_ids, scores=out_scores)
