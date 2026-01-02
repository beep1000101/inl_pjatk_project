from __future__ import annotations

import argparse
import logging

import numpy as np

from src.hybrid.semantic_lsa import (
    _fit_or_load_svd,
    _passages_lsa_path,
)
from src.preprocess.tf_idf_vectors import load_vectorizer as load_tfidf

logger = logging.getLogger(__name__)


def _precompute_passages_lsa(
    *,
    subdataset: str,
    lsa_d: int,
    batch_size: int,
    random_state: int,
    n_iter: int,
) -> str:
    tf = load_tfidf(subdataset)
    X = tf["matrix"]

    svd = _fit_or_load_svd(
        subdataset=subdataset,
        X_passages_tfidf=X,
        n_components=int(lsa_d),
        random_state=int(random_state),
        n_iter=int(n_iter),
    )

    N = int(getattr(X, "shape")[0])
    d = int(getattr(svd, "n_components"))

    out_path = _passages_lsa_path(subdataset, n_components=int(lsa_d))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Precomputing passages LSA: shape=(%s,%s) -> %s",
                N, d, out_path)
    arr = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(N, d),
    )

    bs = int(batch_size)
    for start in range(0, N, bs):
        end = min(start + bs, N)
        Z = svd.transform(X[start:end]).astype(np.float32, copy=False)
        arr[start:end] = Z
        if (start // bs) % 10 == 0 or end == N:
            logger.info("  %s/%s passages", end, N)

    # Ensure data is flushed.
    del arr
    return str(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pretrain (and cache) TruncatedSVD for LSA reranking over cached TF-IDF, "
            "optionally precomputing dense passage embeddings."
        )
    )
    parser.add_argument(
        "--subdataset",
        default="wiki-trivia",
        help="Subdataset name (e.g. wiki-trivia).",
    )
    parser.add_argument(
        "--lsa-d",
        type=int,
        default=256,
        help="LSA dimensionality (TruncatedSVD n_components).",
    )
    parser.add_argument(
        "--svd-n-iter",
        type=int,
        default=5,
        help="TruncatedSVD randomized iterations (fit-time).",
    )
    parser.add_argument(
        "--svd-random-state",
        type=int,
        default=42,
        help="TruncatedSVD random_state (fit-time).",
    )
    parser.add_argument(
        "--precompute-passages",
        action="store_true",
        help="Also precompute dense passage embeddings (faster reranking; larger artifact).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for passage embedding precompute.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    tf = load_tfidf(args.subdataset)
    X = tf["matrix"]
    _ = _fit_or_load_svd(
        subdataset=str(args.subdataset),
        X_passages_tfidf=X,
        n_components=int(args.lsa_d),
        random_state=int(args.svd_random_state),
        n_iter=int(args.svd_n_iter),
    )
    logger.info("SVD ready for subdataset=%s d=%s",
                args.subdataset, int(args.lsa_d))

    if args.precompute_passages:
        out = _precompute_passages_lsa(
            subdataset=str(args.subdataset),
            lsa_d=int(args.lsa_d),
            batch_size=int(args.batch_size),
            random_state=int(args.svd_random_state),
            n_iter=int(args.svd_n_iter),
        )
        logger.info("Precomputed passages embeddings: %s", out)


if __name__ == "__main__":
    main()
