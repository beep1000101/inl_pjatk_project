from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.preprocess.bm25_vectors import load_bm25
from src.preprocess.tf_idf_vectors import load_vectorizer as load_tfidf


@dataclass(frozen=True)
class LexicalCandidates:
    """Lexical candidate set for each query.

    indices: row indices into the lexical passage matrix / passage_ids array
    ids: passage id strings (aligned with indices)
    scores: lexical similarity/score values
    """

    indices: np.ndarray  # (nq, k)
    ids: np.ndarray  # (nq, k) dtype=str
    scores: np.ndarray  # (nq, k) float32


class LexicalRetriever(Protocol):
    passage_ids: np.ndarray

    def retrieve(
        self,
        query_texts: list[str],
        *,
        top_k_candidates: int = 500,
        chunk_size: int = 10_000,
        progress: Any | None = None,
        **kwargs: Any,
    ) -> LexicalCandidates: ...


def _stable_sort_topk(
    *,
    scores: np.ndarray,
    indices: np.ndarray,
    passage_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort each query's top-k deterministically.

    Order: score desc, then passage_id asc.
    """

    nq, k = scores.shape
    out_idx = np.empty_like(indices)
    out_scores = np.empty_like(scores)
    # Avoid `dtype=str` (may truncate strings). Preserve full passage id strings.
    out_ids = np.empty((nq, k), dtype=np.asarray(passage_ids).dtype)

    for i in range(nq):
        idx_row = indices[i]
        scores_row = scores[i]
        pids = np.asarray(passage_ids)[idx_row].astype(str, copy=False)
        order = np.lexsort((pids, -scores_row))
        out_idx[i] = idx_row[order]
        out_scores[i] = scores_row[order]
        out_ids[i] = pids[order]

    return out_idx, out_ids, out_scores


class TfidfLexicalRetriever:
    """TF-IDF lexical retriever (cosine via sparse dot; same as baseline eval)."""

    def __init__(
        self,
        *,
        vectorizer: Any,
        passages_matrix: Any,
        passage_ids: np.ndarray,
    ) -> None:
        self.vectorizer = vectorizer
        self.passages_matrix = passages_matrix
        self.passage_ids = passage_ids

    @classmethod
    def from_cache(cls, subdataset: str) -> "TfidfLexicalRetriever":
        tf = load_tfidf(subdataset)
        return cls(
            vectorizer=tf["vectorizer"],
            passages_matrix=tf["matrix"],
            passage_ids=np.asarray(tf["passage_ids"]),
        )

    def retrieve(
        self,
        query_texts: list[str],
        *,
        top_k_candidates: int = 500,
        chunk_size: int = 10_000,
        progress: Any | None = None,
        **_: Any,
    ) -> LexicalCandidates:
        k = int(top_k_candidates)
        if k <= 0:
            raise ValueError("top_k_candidates must be > 0")

        Q = self.vectorizer.transform(query_texts)  # (nq, D) sparse
        nq = int(Q.shape[0])
        X = self.passages_matrix  # (N, D) sparse
        N = int(X.shape[0])

        top_scores = np.full((nq, k), -np.inf, dtype=np.float32)
        top_idx = np.full((nq, k), 0, dtype=np.int32)

        for start in range(0, N, int(chunk_size)):
            end = min(start + int(chunk_size), N)
            Xb = X[start:end]
            sims = (Xb @ Q.T).toarray().astype(np.float32,
                                               copy=False)  # (B, nq)
            B = int(sims.shape[0])
            if B == 0:
                continue

            k_local = min(k, B)
            idx_local = np.argpartition(-sims,
                                        kth=k_local - 1, axis=0)[:k_local, :]
            scores_local = np.take_along_axis(sims, idx_local, axis=0)
            idx_local = (idx_local + start).astype(np.int32, copy=False)

            comb_scores = np.concatenate([top_scores.T, scores_local], axis=0)
            comb_idx = np.concatenate([top_idx.T, idx_local], axis=0)
            sel = np.argpartition(-comb_scores, kth=k - 1, axis=0)[:k, :]

            top_scores = np.take_along_axis(comb_scores, sel, axis=0).T
            top_idx = np.take_along_axis(comb_idx, sel, axis=0).T

            if progress is not None:
                progress(end, N)

        top_idx, top_ids, top_scores = _stable_sort_topk(
            scores=top_scores, indices=top_idx, passage_ids=self.passage_ids
        )
        return LexicalCandidates(indices=top_idx, ids=top_ids, scores=top_scores)


class Bm25LexicalRetriever:
    """BM25 (Okapi) lexical retriever (same scoring as baseline eval)."""

    def __init__(
        self,
        *,
        vectorizer: Any,
        passages_matrix: Any,
        passage_ids: np.ndarray,
        idf: np.ndarray,
        doc_len: np.ndarray,
        avgdl: float,
    ) -> None:
        self.vectorizer = vectorizer
        self.passages_matrix = passages_matrix
        self.passage_ids = passage_ids
        self.idf = np.asarray(idf, dtype=np.float32)
        self.doc_len = np.asarray(doc_len)
        self.avgdl = float(avgdl) if float(avgdl) > 0 else 1.0

    @classmethod
    def from_cache(cls, subdataset: str) -> "Bm25LexicalRetriever":
        bm = load_bm25(subdataset)
        meta_any = bm.get("meta", {})
        meta = meta_any if isinstance(meta_any, dict) else {}
        avgdl = float(meta.get("avgdl", 0.0))
        return cls(
            vectorizer=bm["vectorizer"],
            passages_matrix=bm["matrix"],
            passage_ids=np.asarray(bm["passage_ids"]),
            idf=np.asarray(bm["idf"]),
            doc_len=np.asarray(bm["doc_len"]),
            avgdl=avgdl,
        )

    def retrieve(
        self,
        query_texts: list[str],
        *,
        top_k_candidates: int = 500,
        chunk_size: int = 10_000,
        progress: Any | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        **_: Any,
    ) -> LexicalCandidates:
        k = int(top_k_candidates)
        if k <= 0:
            raise ValueError("top_k_candidates must be > 0")

        Q = self.vectorizer.transform(query_texts)
        Q = Q.tocsr(copy=True)
        if Q.nnz:
            Q.data[:] = 1.0

        nq = int(Q.shape[0])
        X = self.passages_matrix
        N = int(X.shape[0])
        if int(self.doc_len.shape[0]) != N:
            raise ValueError(
                f"doc_len length must equal N={N}, got {self.doc_len.shape[0]}")

        dl = np.asarray(self.doc_len, dtype=np.float32)

        top_scores = np.full((nq, k), -np.inf, dtype=np.float32)
        top_idx = np.full((nq, k), 0, dtype=np.int32)

        for start in range(0, N, int(chunk_size)):
            end = min(start + int(chunk_size), N)

            Xb = X[start:end].tocsr()
            B = int(Xb.shape[0])
            if B == 0:
                continue

            K = (float(k1) * (1.0 - float(b) + float(b) * (dl[start:end] / self.avgdl))).astype(
                np.float32, copy=False
            )

            Wb = Xb.copy()
            row_nnz = np.diff(Wb.indptr).astype(np.int32, copy=False)
            if Wb.nnz:
                row_idx = np.repeat(np.arange(B, dtype=np.int32), row_nnz)
                tf = Wb.data.astype(np.float32, copy=False)
                denom = tf + K[row_idx]
                tf_norm = (tf * (float(k1) + 1.0)) / np.maximum(denom, 1e-12)
                Wb.data = (tf_norm * self.idf[Wb.indices]
                           ).astype(np.float32, copy=False)

            sims = (Wb @ Q.T).toarray().astype(np.float32,
                                               copy=False)  # (B, nq)
            k_local = min(k, B)
            idx_local = np.argpartition(-sims,
                                        kth=k_local - 1, axis=0)[:k_local, :]
            scores_local = np.take_along_axis(sims, idx_local, axis=0)
            idx_local = (idx_local + start).astype(np.int32, copy=False)

            comb_scores = np.concatenate([top_scores.T, scores_local], axis=0)
            comb_idx = np.concatenate([top_idx.T, idx_local], axis=0)
            sel = np.argpartition(-comb_scores, kth=k - 1, axis=0)[:k, :]

            top_scores = np.take_along_axis(comb_scores, sel, axis=0).T
            top_idx = np.take_along_axis(comb_idx, sel, axis=0).T

            if progress is not None:
                progress(end, N)

        top_idx, top_ids, top_scores = _stable_sort_topk(
            scores=top_scores, indices=top_idx, passage_ids=self.passage_ids
        )
        return LexicalCandidates(indices=top_idx, ids=top_ids, scores=top_scores)
