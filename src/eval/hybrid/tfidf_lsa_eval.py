import argparse
import logging

from src.eval.metrics_io import append_metrics_csv, method_submissions_dir, utc_run_id
from src.eval.retrieval_eval import evaluate_and_write_submission
from src.hybrid.lexical import TfidfLexicalRetriever
from src.hybrid.semantic_lsa import LSATfidfCosineReranker

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hybrid TF-IDF -> LSA (TruncatedSVD over TF-IDF) reranking "
            "on POLEVAL2022 passage retrieval dataset."
        )
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
    parser.add_argument("--k", type=int, default=10,
                        help="Final top-k to output (after reranking).")
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=500,
        help="Lexical candidate set size (TF-IDF stage).",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=100,
        help="How many of the lexical candidates to actually rerank (prefix).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Lexical/semantic fusion weight: final = alpha*lex + (1-alpha)*sem.",
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
        "--fit-svd-if-missing",
        action="store_true",
        help="If pretrained SVD is missing, fit it (slow) and cache under artifacts/lsa/.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000,
        help="Passage chunk size for vectorized lexical scoring (RAM tradeoff).",
    )
    parser.add_argument(
        "--submission-only",
        action="store_true",
        help="Only write TSV (skip metrics against pairs-*.tsv).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    lexical = TfidfLexicalRetriever.from_cache(args.subdataset)
    reranker = LSATfidfCosineReranker.from_cache(
        args.subdataset,
        n_components=int(args.lsa_d),
        random_state=int(args.svd_random_state),
        n_iter=int(args.svd_n_iter),
        fit_if_missing=bool(args.fit_svd_if_missing),
    )

    pairs_split = None if args.submission_only else args.split

    def retriever(texts: list[str], k: int, progress=None):
        candidates = lexical.retrieve(
            texts,
            top_k_candidates=int(args.top_k_candidates),
            chunk_size=int(args.chunk_size),
            progress=progress,
        )
        reranked = reranker.rerank(
            query_texts=texts,
            candidate_indices=candidates.indices,
            candidate_lexical_scores=candidates.scores,
            top_n=int(k),
            rerank_k=min(int(args.rerank_k), int(args.top_k_candidates)),
            alpha=float(args.alpha),
        )
        return reranked.ids

    run_name = "hybrid_tfidf_lsa"
    result = evaluate_and_write_submission(
        dataset_id=args.dataset_id,
        subdataset=args.subdataset,
        questions_split=args.split,
        pairs_split=pairs_split,
        k=int(args.k),
        out_path=None,
        run_name=run_name,
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

    metrics_path = method_submissions_dir(run_name) / "metrics.csv"
    run_id = utc_run_id()
    append_metrics_csv(
        csv_path=metrics_path,
        base_columns=[
            "run_id",
            "method",
            "dataset_id",
            "subdataset",
            "questions_split",
            "pairs_split",
            "k",
            "n_questions",
            "n_labeled",
            "hits_at_k",
            "recall_at_k",
            "precision_at_k",
            "mrr_at_k",
            "ndcg_at_k",
            "out_tsv",
            "submission_only",
            "top_k_candidates",
            "chunk_size",
            "lsa_d",
            "rerank_k",
            "alpha",
            "svd_n_iter",
            "svd_random_state",
        ],
        row={
            "run_id": run_id,
            "method": run_name,
            "dataset_id": args.dataset_id,
            "subdataset": args.subdataset,
            "questions_split": args.split,
            "pairs_split": pairs_split,
            "k": int(args.k),
            "n_questions": int(result.n_questions),
            "n_labeled": int(result.n_labeled) if result.n_labeled is not None else None,
            "hits_at_k": result.hits_at_k,
            "recall_at_k": result.recall_at_k,
            "precision_at_k": result.precision_at_k,
            "mrr_at_k": result.mrr_at_k,
            "ndcg_at_k": result.ndcg_at_k,
            "out_tsv": str(result.out_path),
            "submission_only": bool(args.submission_only),
            "top_k_candidates": int(args.top_k_candidates),
            "chunk_size": int(args.chunk_size),
            "lsa_d": int(args.lsa_d),
            "rerank_k": int(args.rerank_k),
            "alpha": float(args.alpha),
            "svd_n_iter": int(args.svd_n_iter),
            "svd_random_state": int(args.svd_random_state),
        },
    )
    print("Wrote metrics:", metrics_path)


if __name__ == "__main__":
    main()
