from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from src.config.paths import CACHE_DIR
from src.eval.retrieval_eval import (
    load_questions,
    load_relevance_pairs,
    retrieve_bm25_topk,
    retrieve_tfidf_topk,
)
from src.preprocess.bm25_vectors import load_bm25
from src.preprocess.tf_idf_vectors import load_vectorizer as load_tfidf

logger = logging.getLogger(__name__)

MethodName = Literal["bm25", "tfidf"]
KSchedule = Literal["linear", "log"]


@dataclass(frozen=True)
class MethodCurve:
    method: MethodName
    k_values: np.ndarray  # (P,)
    hits_values: np.ndarray  # (P,)
    hits_at_max_k: float
    k_90pct_of_max: int | None


def _first_relevant_ranks(
    *,
    preds: np.ndarray,
    qids: np.ndarray,
    rel_by_q: dict[int, set[str]],
) -> tuple[np.ndarray, int]:
    """Return first-relevant rank per labeled question.

    ranks are 1..K, or K+1 if no relevant in top-K.
    """

    if preds.ndim != 2:
        raise ValueError(f"preds must be 2D, got shape={preds.shape}")

    K = int(preds.shape[1])
    first_ranks: list[int] = []

    for row_idx, qid in enumerate(qids.tolist()):
        rel = rel_by_q.get(int(qid))
        if not rel:
            continue

        ranked = preds[row_idx]
        first = K + 1
        for r in range(K):
            pid = str(ranked[r])
            if pid and pid in rel:
                first = r + 1
                break
        first_ranks.append(first)

    arr = np.asarray(first_ranks, dtype=np.int32)
    return arr, int(arr.shape[0])


def _hits_curve_from_first_ranks(first_ranks: np.ndarray, *, max_k: int) -> np.ndarray:
    if first_ranks.ndim != 1:
        raise ValueError("first_ranks must be 1D")
    if max_k <= 0:
        raise ValueError("max_k must be > 0")
    if first_ranks.size == 0:
        return np.full((max_k,), np.nan, dtype=np.float32)

    # first_ranks values are 1..max_k+1; treat max_k+1 as "no hit".
    clipped = np.clip(first_ranks, 1, max_k + 1)
    counts = np.bincount(clipped, minlength=max_k + 2).astype(np.int64, copy=False)

    # hits@k is cumulative count of ranks <= k divided by n_labeled
    cum_hits = np.cumsum(counts[1 : max_k + 1])
    return (cum_hits / float(first_ranks.size)).astype(np.float32)


def _hits_at_ks_from_first_ranks(first_ranks: np.ndarray, *, ks: np.ndarray) -> np.ndarray:
    if first_ranks.ndim != 1:
        raise ValueError("first_ranks must be 1D")
    if ks.ndim != 1:
        raise ValueError("ks must be 1D")
    if first_ranks.size == 0:
        return np.full((int(ks.size),), np.nan, dtype=np.float32)
    return np.asarray([(first_ranks <= int(k)).mean() for k in ks], dtype=np.float32)


def _k_at_fraction_of_max(hits: np.ndarray, *, frac: float) -> int | None:
    if hits.size == 0 or not np.isfinite(hits[-1]):
        return None
    target = float(frac) * float(hits[-1])
    idx = np.where(hits >= target)[0]
    if idx.size == 0:
        return None
    return int(idx[0] + 1)  # k is 1-indexed


def _k_at_fraction_of_max_from_points(ks: np.ndarray, hits: np.ndarray, *, frac: float) -> int | None:
    if ks.size == 0 or hits.size == 0:
        return None
    if ks.shape != hits.shape:
        raise ValueError("ks and hits must have same shape")
    if not np.isfinite(hits).any():
        return None
    hits_max = float(hits[-1])
    if not np.isfinite(hits_max):
        return None
    target = float(frac) * hits_max
    idx = np.where(hits >= target)[0]
    if idx.size == 0:
        return None
    return int(ks[int(idx[0])])


def make_k_values(*, schedule: KSchedule, max_k: int, points: int, min_k: int = 1) -> np.ndarray:
    if max_k <= 0:
        raise ValueError("max_k must be > 0")
    min_k_i = max(1, int(min_k))
    max_k_i = int(max_k)
    if min_k_i > max_k_i:
        min_k_i = 1

    if schedule == "linear":
        # still keep it downsampled to `points`
        p = max(2, int(points))
        ks = np.unique(np.round(np.linspace(min_k_i, max_k_i, num=p)).astype(np.int32))
    elif schedule == "log":
        # "superlinear" progression: dense at small k, sparse at large k
        p = max(3, int(points))
        if min_k_i == max_k_i:
            ks = np.asarray([max_k_i], dtype=np.int32)
        else:
            ks = np.unique(
                np.round(
                    np.logspace(
                        np.log10(float(min_k_i)),
                        np.log10(float(max_k_i)),
                        num=p,
                    )
                ).astype(np.int32)
            )
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    ks = np.clip(ks, 1, max_k_i)
    # Ensure endpoints exist.
    if ks.size == 0 or int(ks[0]) != min_k_i:
        ks = np.unique(np.concatenate([np.asarray([min_k_i], dtype=np.int32), ks]))
    if int(ks[-1]) != max_k_i:
        ks = np.unique(np.concatenate([ks, np.asarray([max_k_i], dtype=np.int32)]))
    return ks.astype(np.int32, copy=False)


def _retrieve_max_k(
    *,
    method: MethodName,
    dataset_id: str,
    subdataset: str,
    split: str,
    max_k: int,
    bm25_k1: float,
    bm25_b: float,
    bm25_chunk_size: int,
    tfidf_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    questions_df = load_questions(dataset_id, subdataset, split)
    qids = questions_df.index.to_numpy(dtype=int)
    q_texts = questions_df["text"].tolist()

    logger.info("Loaded %s questions", len(q_texts))

    def progress(done: int, total: int) -> None:
        if total <= 0:
            return
        pct = 100.0 * float(done) / float(total)
        logger.info("Retrieval progress: %s/%s (%.1f%%)", done, total, pct)

    if method == "tfidf":
        tf = cast(dict[str, Any], load_tfidf(subdataset))
        preds = retrieve_tfidf_topk(
            vectorizer=tf["vectorizer"],
            passages_matrix=tf["matrix"],
            passage_ids=cast(np.ndarray, tf["passage_ids"]),
            query_texts=q_texts,
            k=int(max_k),
            chunk_size=int(tfidf_chunk_size),
            progress=progress,
        )
        return preds, qids

    if method == "bm25":
        bm = cast(dict[str, Any], load_bm25(subdataset))
        meta = cast(dict[str, Any], bm.get("meta", {}))
        avgdl = float(meta.get("avgdl", 0.0))
        preds = retrieve_bm25_topk(
            vectorizer=bm["vectorizer"],
            passages_matrix=bm["matrix"],
            passage_ids=cast(np.ndarray, bm["passage_ids"]),
            idf=cast(np.ndarray, bm["idf"]),
            doc_len=cast(np.ndarray, bm["doc_len"]),
            avgdl=float(avgdl),
            query_texts=q_texts,
            k=int(max_k),
            k1=float(bm25_k1),
            b=float(bm25_b),
            chunk_size=int(bm25_chunk_size),
            progress=progress,
        )
        return preds, qids

    raise ValueError(f"Unknown method: {method}")


def compute_hits_curve(
    *,
    method: MethodName,
    dataset_id: str,
    subdataset: str,
    split: str,
    max_k: int,
    ks: np.ndarray,
    frac_of_max: float,
    bm25_k1: float,
    bm25_b: float,
    bm25_chunk_size: int,
    tfidf_chunk_size: int,
    dense_curve: bool,
) -> MethodCurve:
    rel_by_q = load_relevance_pairs(dataset_id, subdataset, split=split)
    logger.info("Loaded relevance labels for %s questions", len(rel_by_q))

    preds, qids = _retrieve_max_k(
        method=method,
        dataset_id=dataset_id,
        subdataset=subdataset,
        split=split,
        max_k=int(max_k),
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
        bm25_chunk_size=int(bm25_chunk_size),
        tfidf_chunk_size=int(tfidf_chunk_size),
    )

    preds = np.asarray(preds, dtype=object)
    if preds.ndim != 2 or preds.shape[1] != int(max_k):
        raise ValueError(f"expected preds shape (nq,{max_k}), got {preds.shape}")

    first_ranks, n_labeled = _first_relevant_ranks(preds=preds, qids=qids, rel_by_q=rel_by_q)
    if n_labeled == 0:
        logger.warning("No labeled questions found for split=%s; hits@k will be NaN", split)

    max_k_i = int(max_k)
    ks_i = np.asarray(ks, dtype=np.int32)
    if ks_i.ndim != 1 or ks_i.size == 0:
        raise ValueError("ks must be a non-empty 1D array")
    if int(ks_i[-1]) != max_k_i:
        raise ValueError("ks must include max_k as the last element")

    if bool(dense_curve):
        # Full 1..max_k curve (can be large if max_k is huge)
        hits = _hits_curve_from_first_ranks(first_ranks, max_k=max_k_i)
        k90 = _k_at_fraction_of_max(hits, frac=float(frac_of_max))
        k_values = np.arange(1, max_k_i + 1, dtype=np.int32)
        hits_values = hits
    else:
        hits_values = _hits_at_ks_from_first_ranks(first_ranks, ks=ks_i)
        k90 = _k_at_fraction_of_max_from_points(ks_i, hits_values, frac=float(frac_of_max))
        k_values = ks_i

    return MethodCurve(
        method=method,
        k_values=k_values,
        hits_values=hits_values,
        hits_at_max_k=float(hits_values[-1]) if np.isfinite(hits_values[-1]) else float("nan"),
        k_90pct_of_max=k90,
    )


def _plot_curves(
    *,
    curves: list[MethodCurve],
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. Install it (in the venv): "
            "python -m pip install matplotlib"
        ) from e

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for c in curves:
        plt.plot(c.k_values, c.hits_values, label=c.method)
        if c.k_90pct_of_max is not None:
            plt.axvline(c.k_90pct_of_max, linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Hits@k")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate top-k by computing Hits@k curves for BM25 and TF-IDF. "
            "Retrieves once at max-k and derives all smaller k from that run."
        )
    )
    parser.add_argument(
        "--dataset-id",
        default="piotr-rybak__poleval2022-passage-retrieval-dataset",
    )
    parser.add_argument("--subdataset", default="wiki-trivia")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--max-k", type=int, default=1000)
    parser.add_argument(
        "--frac-of-max",
        type=float,
        default=0.9,
        help="Find smallest k with Hits@k >= frac * Hits@max_k.",
    )
    parser.add_argument(
        "--k-schedule",
        default="log",
        choices=["linear", "log"],
        help="How to choose k values between 1 and max-k (log is superlinear / log-spaced).",
    )
    parser.add_argument(
        "--k-points",
        type=int,
        default=50,
        help="Number of k points to evaluate for the chosen schedule.",
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=1,
        help="Smallest k to include (still ensures max-k is included).",
    )
    parser.add_argument(
        "--dense-curve",
        action="store_true",
        help="Also compute and write the full dense curve for k=1..max-k.",
    )
    parser.add_argument(
        "--methods",
        default="bm25,tfidf",
        help="Comma-separated: bm25,tfidf",
    )

    # BM25 knobs
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--bm25-chunk-size", type=int, default=10_000)

    # TF-IDF knob
    parser.add_argument("--tfidf-chunk-size", type=int, default=10_000)

    # Output
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: .cache/calibration/<subdataset>/<split>/)",
    )
    parser.add_argument("--plot", action="store_true", help="Also write a PNG plot.")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    methods = [m.strip() for m in str(args.methods).split(",") if m.strip()]
    bad = [m for m in methods if m not in {"bm25", "tfidf"}]
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Supported: bm25,tfidf")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else (CACHE_DIR / "calibration" / str(args.subdataset) / str(args.split))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    ks = make_k_values(
        schedule=cast(KSchedule, str(args.k_schedule)),
        max_k=int(args.max_k),
        points=int(args.k_points),
        min_k=int(args.min_k),
    )

    curves: list[MethodCurve] = []
    for method in methods:
        logger.info("Computing hits curve for method=%s", method)
        curve = compute_hits_curve(
            method=cast(MethodName, method),
            dataset_id=str(args.dataset_id),
            subdataset=str(args.subdataset),
            split=str(args.split),
            max_k=int(args.max_k),
            ks=ks,
            frac_of_max=float(args.frac_of_max),
            bm25_k1=float(args.bm25_k1),
            bm25_b=float(args.bm25_b),
            bm25_chunk_size=int(args.bm25_chunk_size),
            tfidf_chunk_size=int(args.tfidf_chunk_size),
            dense_curve=bool(args.dense_curve),
        )
        curves.append(curve)

        if curve.k_90pct_of_max is not None:
            logger.info(
                "%s: Hits@%s=%.4f; smallest k for %.0f%% of max is %s",
                curve.method,
                int(args.max_k),
                curve.hits_at_max_k,
                100.0 * float(args.frac_of_max),
                curve.k_90pct_of_max,
            )

    # Write tidy "data" CSV: one row per (method, k)
    rows: list[dict[str, object]] = []
    for c in curves:
        for k, h in zip(c.k_values.tolist(), c.hits_values.tolist(), strict=True):
            rows.append(
                {
                    "method": c.method,
                    "k": int(k),
                    "hits_at_k": float(h) if h is not None else None,
                    "max_k": int(args.max_k),
                    "k_schedule": str(args.k_schedule),
                    "k_points": int(args.k_points),
                    "frac_of_max": float(args.frac_of_max),
                }
            )
    points_df = pd.DataFrame(rows)

    summary_rows = []
    for c in curves:
        summary_rows.append(
            {
                "method": c.method,
                "max_k": int(args.max_k),
                "hits_at_max_k": c.hits_at_max_k,
                "frac_of_max": float(args.frac_of_max),
                "k_at_frac_of_max": c.k_90pct_of_max,
            }
        )

    stem = f"maxk{int(args.max_k)}_{str(args.k_schedule)}_p{int(args.k_points)}"
    points_csv = out_dir / f"hits_points_{stem}.csv"
    summary_csv = out_dir / f"hits_summary_{stem}.csv"

    points_df.to_csv(points_csv, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    print("Wrote:", points_csv)
    print("Wrote:", summary_csv)

    if args.dense_curve:
        # Dense wide curve is useful for quick plotting in notebooks.
        wide = pd.DataFrame({"k": np.arange(1, int(args.max_k) + 1, dtype=np.int32)})
        for c in curves:
            wide[f"hits_{c.method}"] = c.hits_values
        dense_csv = out_dir / f"hits_curve_dense_{stem}.csv"
        wide.to_csv(dense_csv, index=False)
        print("Wrote:", dense_csv)

    if args.plot:
        plot_path = out_dir / f"hits_curve_{stem}.png"
        title = f"Hits@k vs k ({args.subdataset}, split={args.split}, max_k={int(args.max_k)})"
        _plot_curves(curves=curves, out_path=plot_path, title=title)
        print("Wrote:", plot_path)


if __name__ == "__main__":
    main()
