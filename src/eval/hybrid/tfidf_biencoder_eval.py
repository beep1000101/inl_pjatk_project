import argparse
import logging

from src.eval.metrics_io import append_metrics_csv, method_submissions_dir, utc_run_id
from src.eval.retrieval_eval import evaluate_and_write_submission
from src.hybrid.lexical import TfidfLexicalRetriever
from src.hybrid.semantic_biencoder import BiEncoderCosineReranker

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str | None:
    d = str(device).strip().lower()
    if d in {"", "auto"}:
        return None
    return str(device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hybrid TF-IDF -> bi-encoder (Sentence-Transformers) cosine reranking "
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
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Final top-k to output (after reranking).",
    )
    parser.add_argument(
        "--top-k-candidates",
        type=int,
        default=500,
        help="Lexical candidate set size (TF-IDF stage).",
    )
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=500,
        help="How many of the lexical candidates to actually rerank (prefix).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help=(
            "Optional lexical/semantic fusion weight in [0,1]. "
            "If omitted, ranks by semantic score only."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-Transformers bi-encoder model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help='Device for embeddings: "cpu", "cuda", or "auto".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length for encoder truncation.",
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
    reranker = BiEncoderCosineReranker.from_dataset_cache(
        dataset_id=args.dataset_id,
        subdataset=args.subdataset,
        model_name=args.model_name,
        device=_resolve_device(args.device),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
    )

    pairs_split = None if args.submission_only else args.split

    def retriever(texts: list[str], k: int, progress=None):
        candidates = lexical.retrieve(
            texts,
            top_k_candidates=int(args.top_k_candidates),
            chunk_size=int(args.chunk_size),
            progress=progress,
        )
        rk = min(int(args.rerank_k), int(args.top_k_candidates))
        reranked = reranker.rerank(
            query_texts=texts,
            candidate_indices=candidates.indices,
            candidate_lexical_scores=candidates.scores,
            top_n=int(k),
            rerank_k=rk,
            alpha=args.alpha,
        )
        return reranked.ids

    run_name = "hybrid_tfidf_biencoder"
    run_id = utc_run_id()
    out_path = (
        method_submissions_dir(run_name)
        / f"{args.subdataset}_questions-{args.split}__{run_id}.tsv"
    )

    result = evaluate_and_write_submission(
        dataset_id=args.dataset_id,
        subdataset=args.subdataset,
        questions_split=args.split,
        pairs_split=pairs_split,
        k=int(args.k),
        out_path=out_path,
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
    used_device = reranker.device

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
            "rerank_k",
            "alpha",
            "biencoder_model",
            "biencoder_device",
            "biencoder_batch_size",
            "biencoder_max_length",
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
            "rerank_k": int(args.rerank_k),
            "alpha": float(args.alpha) if args.alpha is not None else None,
            "biencoder_model": str(args.model_name),
            "biencoder_device": str(used_device),
            "biencoder_batch_size": int(args.batch_size),
            "biencoder_max_length": int(args.max_length),
        },
    )
    print("Wrote metrics:", metrics_path)


if __name__ == "__main__":
    main()
