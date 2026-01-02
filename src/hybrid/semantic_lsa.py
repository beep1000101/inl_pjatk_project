from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Any, cast

import joblib
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.config.paths import REPO_ROOT
from src.hybrid.reranker import RerankResult, minmax01
from src.preprocess.tf_idf_vectors import load_vectorizer as load_tfidf

logger = logging.getLogger(__name__)


def _artifacts_dir(subdataset: str) -> Path:
    return REPO_ROOT / "artifacts" / "lsa" / str(subdataset)


def _svd_path(subdataset: str, *, n_components: int) -> Path:
    return _artifacts_dir(subdataset) / f"svd_d{int(n_components)}.joblib"


def _passages_lsa_path(subdataset: str, *, n_components: int) -> Path:
    return _artifacts_dir(subdataset) / f"passages_lsa_d{int(n_components)}.f32.npy"


@dataclass(frozen=True)
class SvdMeta:
    subdataset: str
    n_components: int
    n_passages: int
    n_features: int


def _fit_or_load_svd(
    *,
    subdataset: str,
    X_passages_tfidf: object,
    n_components: int,
    random_state: int,
    n_iter: int,
) -> TruncatedSVD:
    path = _svd_path(subdataset, n_components=n_components)
    if path.is_file():
        obj = joblib.load(path)
        if isinstance(obj, dict) and "svd" in obj:
            return obj["svd"]
        if isinstance(obj, TruncatedSVD):
            return obj
        raise ValueError(f"Unexpected SVD artifact format in: {path}")

    X = cast(sparse.spmatrix, X_passages_tfidf)
    n_passages = int(X.shape[0])
    n_features = int(X.shape[1])

    # TruncatedSVD requires 1 <= n_components < n_features.
    d = int(n_components)
    d = min(d, max(1, n_features - 1))
    d = min(d, max(1, n_passages - 1))
    if d <= 0:
        raise ValueError(
            f"Cannot fit SVD: invalid effective components d={d} for shape={getattr(X, 'shape', None)}"
        )
    if d != int(n_components):
        logger.warning(
            "Adjusting n_components from %s to %s for subdataset=%s (n_passages=%s, n_features=%s)",
            n_components,
            d,
            subdataset,
            n_passages,
            n_features,
        )

    logger.info(
        "Fitting TruncatedSVD(d=%s) for subdataset=%s on TF-IDF matrix shape=%s",
        d,
        subdataset,
        getattr(X, "shape", None),
    )

    svd = TruncatedSVD(
        n_components=d,
        algorithm="randomized",
        n_iter=int(n_iter),
        random_state=int(random_state),
    )
    svd.fit(X)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "svd": svd,
            "meta": SvdMeta(
                subdataset=str(subdataset),
                n_components=int(d),
                n_passages=int(n_passages),
                n_features=int(n_features),
            ),
        },
        path,
    )
    logger.info("Saved SVD artifact: %s", path)
    return svd


def _cosine_scores_1vM(*, query_vec: np.ndarray, cand_vecs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
    x = np.asarray(cand_vecs, dtype=np.float32)

    qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), eps)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return (xn @ qn.T).ravel().astype(np.float32, copy=False)


class LSATfidfCosineReranker:
    """LSA reranker: TruncatedSVD over cached TF-IDF + cosine similarity.

    Notes:
    - Uses cached TF-IDF artifacts from `.cache/preprocessed_data/tf_idf_vectors/<subdataset>/`.
    - Caches only the SVD model under `artifacts/lsa/<subdataset>/` (outside `.cache`).
    - Scores only lexical candidate rows (rerank-only), not the whole corpus.
    """

    def __init__(
        self,
        *,
        subdataset: str,
        vectorizer: Any,
        passages_tfidf: sparse.spmatrix,
        passage_ids: np.ndarray,
        svd: TruncatedSVD,
        passages_lsa: np.ndarray | None = None,
    ) -> None:
        self.subdataset = str(subdataset)
        self.vectorizer = vectorizer
        self.passages_tfidf = passages_tfidf
        self.passage_ids = np.asarray(passage_ids)
        self.svd = svd
        self.passages_lsa = passages_lsa

    @classmethod
    def from_cache(
        cls,
        subdataset: str,
        *,
        n_components: int = 256,
        random_state: int = 42,
        n_iter: int = 5,
        fit_if_missing: bool = False,
        use_precomputed_passages: bool = True,
    ) -> "LSATfidfCosineReranker":
        tf = load_tfidf(subdataset)
        X = cast(sparse.spmatrix, tf["matrix"])

        svd_path = _svd_path(str(subdataset), n_components=int(n_components))
        if not svd_path.is_file() and not bool(fit_if_missing):
            raise FileNotFoundError(
                "Pretrained SVD artifact not found: "
                f"{svd_path}. Run: python -m src.preprocess.lsa_svd "
                f"--subdataset {subdataset} --lsa-d {int(n_components)}"
            )

        svd = _fit_or_load_svd(
            subdataset=str(subdataset),
            X_passages_tfidf=X,
            n_components=int(n_components),
            random_state=int(random_state),
            n_iter=int(n_iter),
        )

        passages_lsa = None
        if use_precomputed_passages:
            p_path = _passages_lsa_path(
                str(subdataset), n_components=int(n_components))
            if p_path.is_file():
                passages_lsa = np.load(p_path, mmap_mode="r")

        return cls(
            subdataset=str(subdataset),
            vectorizer=tf["vectorizer"],
            passages_tfidf=X,
            passage_ids=np.asarray(tf["passage_ids"]),
            svd=svd,
            passages_lsa=passages_lsa,
        )

    def rerank(
        self,
        *,
        query_texts: list[str],
        candidate_indices: np.ndarray,
        candidate_lexical_scores: np.ndarray,
        top_n: int = 10,
        rerank_k: int | None = None,
        alpha: float | None = None,
    ) -> RerankResult:
        top_n = int(top_n)
        if top_n <= 0:
            raise ValueError("top_n must be > 0")

        cand_idx = np.asarray(candidate_indices, dtype=np.int32)
        cand_lex = np.asarray(candidate_lexical_scores, dtype=np.float32)
        nq, K = cand_idx.shape

        if len(query_texts) != nq:
            raise ValueError(
                f"query_texts length must match candidate_indices rows (nq={nq}), got {len(query_texts)}"
            )

        rk = int(rerank_k) if rerank_k is not None else int(K)
        if rk <= 0:
            raise ValueError("rerank_k must be > 0")
        rk = min(rk, int(K))
        if top_n > rk:
            raise ValueError(
                "top_n must be <= rerank_k (rerank prefix must cover outputs)")

        Q = self.vectorizer.transform([str(t) for t in query_texts])
        Qz = self.svd.transform(Q).astype(np.float32, copy=False)

        out_idx = np.empty((nq, top_n), dtype=np.int32)
        out_scores = np.empty((nq, top_n), dtype=np.float32)
        out_ids = np.empty(
            (nq, top_n), dtype=np.asarray(self.passage_ids).dtype)

        for i in range(nq):
            idx_row = cand_idx[i][:rk]
            lex_row = cand_lex[i][:rk]

            if self.passages_lsa is not None:
                Z_cand = np.asarray(
                    self.passages_lsa[idx_row], dtype=np.float32)
            else:
                X_cand = cast(Any, self.passages_tfidf)[idx_row]
                Z_cand = self.svd.transform(
                    X_cand).astype(np.float32, copy=False)

            sem_scores = _cosine_scores_1vM(query_vec=Qz[i], cand_vecs=Z_cand)
            pids = np.asarray(self.passage_ids)[
                idx_row].astype(str, copy=False)

            if alpha is None:
                order = np.lexsort((pids, -lex_row, -sem_scores))
            else:
                a = float(alpha)
                if not (0.0 <= a <= 1.0):
                    raise ValueError("alpha must be in [0, 1]")
                fused = (a * minmax01(lex_row) + (1.0 - a) * minmax01(sem_scores)).astype(
                    np.float32, copy=False
                )
                order = np.lexsort((pids, -sem_scores, -fused))

            order = order[:top_n]
            out_idx[i] = idx_row[order]
            out_scores[i] = sem_scores[order]
            out_ids[i] = pids[order]

        return RerankResult(indices=out_idx, ids=out_ids, scores=out_scores)
