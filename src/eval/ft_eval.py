import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config.paths import CACHE_DIR
from src.eval.retrieval_eval import (
    build_faiss_flat_ip_index,
    build_faiss_ivfpq_ip_index,
    evaluate_and_write_submission,
    embed_fasttext_avg,
    retrieve_dense_faiss_topk,
)
from src.models.fasttext import FastTextModelSpec, load_fasttext_model
from src.preprocess.fasttext_vectors import load_fasttext_vectors, tokenize

logger = logging.getLogger(__name__)


def _default_index_path(subdataset: str, index_type: str) -> Path:
    name = "faiss_flat_ip.index" if index_type == "flat" else "faiss_ivfpq_ip.index"
    return CACHE_DIR / "preprocessed_data" / "fasttext_vectors" / subdataset / name


def _load_or_build_index(
    *,
    vectors: np.ndarray,
    index_path: Path | None,
    index_type: str,
    nlist: int,
    m: int,
    nbits: int,
    train_size: int,
    chunk_size: int,
    nprobe: int,
) -> Any:
    try:
        import faiss  # type: ignore
    except Exception as e:
        raise ImportError(
            "faiss is required. Ensure `faiss-cpu` is installed in your environment."
        ) from e

    if index_path is not None and index_path.is_file():
        logger.info("Loading FAISS index from: %s", index_path)
        index = faiss.read_index(str(index_path))
        if hasattr(index, "nprobe"):
            index.nprobe = int(nprobe)
        return index

    if index_type == "flat":
        logger.info(
            "Building FAISS IndexFlatIP (cosine via NumPy normalization)")
        index = build_faiss_flat_ip_index(
            passage_vectors=vectors,
            chunk_size=chunk_size,
        )
    elif index_type == "ivfpq":
        logger.info(
            "Building FAISS IVF-PQ index (this may take a while): nlist=%s m=%s nbits=%s train=%s",
            nlist,
            m,
            nbits,
            train_size,
        )

        def progress(done: int, total: int) -> None:
            if total > 0 and (done == total or done % (1_000_000) == 0):
                logger.info("Index add progress: %s/%s", done, total)

        index = build_faiss_ivfpq_ip_index(
            passage_vectors=vectors,
            nlist=nlist,
            m=m,
            nbits=nbits,
            train_size=train_size,
            chunk_size=chunk_size,
            progress=progress,
        )
        index.nprobe = int(nprobe)
    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    if index_path is not None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving FAISS index to: %s", index_path)
        faiss.write_index(index, str(index_path))

    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FastText passage embeddings with FAISS retrieval (cosine) on POLEVAL2022."  # noqa: E501
    )
    parser.add_argument(
        "--dataset-id",
        default="piotr-rybak__poleval2022-passage-retrieval-dataset",
        help="HF dataset id used by the project cache.",
    )
    parser.add_argument(
        "--subdataset",
        default="wiki-trivia",
        help="Subdataset name (e.g. wiki-trivia).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "test"],
        help="Questions split to retrieve for.",
    )
    parser.add_argument("--k", type=int, default=10, help="Top-k to retrieve.")
    parser.add_argument(
        "--submission-only",
        action="store_true",
        help="Only write TSV (skip metrics against pairs-*.tsv).",
    )

    # FAISS params (kept minimal)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Optional path to save/load the FAISS index (disabled if omitted).",
    )
    parser.add_argument(
        "--index-type",
        choices=["flat", "ivfpq"],
        default="flat",
        help="FAISS index type. flat=exact (more RAM), ivfpq=compressed (train required).",
    )
    parser.add_argument("--nprobe", type=int, default=32, help="IVF nprobe.")
    parser.add_argument("--nlist", type=int, default=4096, help="IVF nlist.")
    parser.add_argument("--m", type=int, default=30, help="PQ m.")
    parser.add_argument("--nbits", type=int, default=8, help="PQ nbits.")
    parser.add_argument(
        "--train-size",
        type=int,
        default=200_000,
        help="Training sample size for IVF-PQ.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Add vectors to FAISS in chunks.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    ft = load_fasttext_vectors(args.subdataset)
    vectors = ft["vectors"]
    passage_ids = ft["passage_ids"]

    assert isinstance(vectors, np.ndarray)
    assert isinstance(passage_ids, np.ndarray)

    n, d = vectors.shape
    logger.info("Loaded vectors: shape=%s dtype=%s", (n, d), vectors.dtype)

    model = load_fasttext_model(spec=FastTextModelSpec(lang="pl", dim=300))

    index_path: Path | None = args.index_path
    # If user didn't specify an index path, do not save (and do not auto-load).
    # You can pass --index-path to reuse the built index across runs.
    if index_path is not None and index_path.is_dir():
        index_path = index_path / \
            _default_index_path(args.subdataset, args.index_type).name

    index = _load_or_build_index(
        vectors=vectors,
        index_path=index_path,
        index_type=args.index_type,
        nlist=args.nlist,
        m=args.m,
        nbits=args.nbits,
        train_size=args.train_size,
        chunk_size=args.chunk_size,
        nprobe=args.nprobe,
    )

    def retriever(texts: list[str], k: int) -> np.ndarray:
        Q = embed_fasttext_avg(model=model, texts=texts, tokenize=tokenize)
        return retrieve_dense_faiss_topk(
            index=index,
            passage_ids=passage_ids,
            query_vectors=Q,
            k=k,
        )

    pairs_split = None if args.submission_only else args.split

    result = evaluate_and_write_submission(
        dataset_id=args.dataset_id,
        subdataset=args.subdataset,
        questions_split=args.split,
        pairs_split=pairs_split,
        k=int(args.k),
        out_path=None,
        run_name="fasttext_faiss",
        retriever=retriever,
    )

    print("Wrote:", result.out_path)
    if result.hits_at_k is not None:
        k = int(args.k)
        print(f"Hits@{k}:      {result.hits_at_k:.4f}")
        print(f"Recall@{k}:    {result.recall_at_k:.4f}")
        print(f"Precision@{k}: {result.precision_at_k:.4f}")
        print(f"MRR@{k}:       {result.mrr_at_k:.4f}")
        print(f"nDCG@{k}:      {result.ndcg_at_k:.4f}")


if __name__ == "__main__":
    main()
