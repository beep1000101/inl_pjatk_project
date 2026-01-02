import argparse
import logging
from typing import Any, cast

import numpy as np

from src.eval.metrics_io import append_metrics_csv, method_submissions_dir, utc_run_id
from src.eval.retrieval_eval import evaluate_and_write_submission, retrieve_tfidf_topk
from src.preprocess.tf_idf_vectors import load_vectorizer as load_tfidf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TF-IDF + cosine similarity retrieval on POLEVAL2022 passage retrieval dataset."
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
        "--chunk-size",
        type=int,
        default=10_000,
        help="Passage chunk size for vectorized similarity computation (RAM tradeoff).",
    )
    parser.add_argument(
        "--submission-only",
        action="store_true",
        help="Only write TSV (skip metrics against pairs-*.tsv).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    tf = cast(dict[str, Any], load_tfidf(args.subdataset))
    vectorizer = tf["vectorizer"]
    passages_matrix = tf["matrix"]
    passage_ids = cast(np.ndarray, tf["passage_ids"])
    meta = cast(dict[str, Any], tf.get("meta", {}))

    pairs_split = None if args.submission_only else args.split

    def retriever(texts: list[str], k: int, progress=None) -> np.ndarray:
        return retrieve_tfidf_topk(
            vectorizer=vectorizer,
            passages_matrix=passages_matrix,
            passage_ids=passage_ids,
            query_texts=texts,
            k=k,
            chunk_size=int(args.chunk_size),
            progress=progress,
        )

    result = evaluate_and_write_submission(
        dataset_id=args.dataset_id,
        subdataset=args.subdataset,
        questions_split=args.split,
        pairs_split=pairs_split,
        k=int(args.k),
        out_path=None,
        run_name="tfidf_cosine",
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

    metrics_path = method_submissions_dir("tfidf_cosine") / "metrics.csv"
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
            "chunk_size",
            "tfidf_out_dir",
            "tfidf_n_passages",
            "tfidf_n_features",
            "tfidf_min_df",
            "tfidf_max_df",
            "tfidf_max_features",
        ],
        row={
            "run_id": run_id,
            "method": "tfidf_cosine",
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
            "chunk_size": int(args.chunk_size),
            "tfidf_out_dir": str(tf.get("out_dir")) if tf.get("out_dir") is not None else None,
            "tfidf_n_passages": meta.get("n_passages"),
            "tfidf_n_features": meta.get("n_features"),
            "tfidf_min_df": meta.get("min_df"),
            "tfidf_max_df": meta.get("max_df"),
            "tfidf_max_features": meta.get("max_features"),
        },
    )
    print("Wrote metrics:", metrics_path)


if __name__ == "__main__":
    main()
