from __future__ import annotations

import logging
from dataclasses import dataclass
import time

import numpy as np
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from src.config.paths import poleval2022_passages_path
from src.hybrid.reranker import RerankResult, minmax01
from src.preprocess.tf_idf_vectors import drop_redirects, join_title_and_text, read_jsonl

logger = logging.getLogger(__name__)


def _default_device(device: str | None) -> str:
    if device is None:
        return "cuda" if _torch().cuda.is_available() else "cpu"
    d = str(device).strip().lower()
    if d in {"auto", ""}:
        return "cuda" if _torch().cuda.is_available() else "cpu"
    return str(device)


def _torch():
    import torch

    return torch


def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer


@dataclass(frozen=True)
class _Passages:
    ids: np.ndarray
    texts: list[str]


def _load_passages_from_dataset_cache(*, dataset_id: str, subdataset: str) -> _Passages:
    passages_path = poleval2022_passages_path(dataset_id, subdataset)
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages file not found in dataset cache: {passages_path}. "
            "Make sure the dataset cache is populated under .cache/data/."
        )

    t0 = time.monotonic()
    logger.info("Loading passages JSONL: %s", passages_path)
    df = read_jsonl(passages_path)
    raw_n = int(df.shape[0])
    logger.info("Loaded %s raw passage rows in %.1fs",
                raw_n, time.monotonic() - t0)

    t1 = time.monotonic()
    df = drop_redirects(df).reset_index(drop=True)
    kept_n = int(df.shape[0])
    logger.info(
        "After drop_redirects: %s rows (filtered %s) in %.1fs",
        kept_n,
        raw_n - kept_n,
        time.monotonic() - t1,
    )

    t2 = time.monotonic()
    texts = join_title_and_text(df).astype(str).tolist()
    ids_list = df["id"].astype(str).tolist()
    logger.info("Prepared passage texts/ids in %.1fs", time.monotonic() - t2)

    return _Passages(ids=np.asarray(ids_list, dtype=str), texts=texts)


def _verify_alignment_with_lexical_caches(*, subdataset: str, passage_ids: np.ndarray) -> None:
    expected: np.ndarray | None = None

    try:
        from src.preprocess.tf_idf_vectors import load_vectorizer as load_tfidf

        expected = np.asarray(load_tfidf(subdataset)["passage_ids"])
        source = "tf_idf_vectors"
    except FileNotFoundError:
        expected = None
        source = ""

    if expected is None:
        try:
            from src.preprocess.bm25_vectors import load_bm25

            expected = np.asarray(load_bm25(subdataset)["passage_ids"])
            source = "bm25_vectors"
        except FileNotFoundError:
            expected = None
            source = ""

    if expected is None:
        logger.warning(
            "Could not verify passage_id alignment for subdataset=%s (no lexical caches found)",
            subdataset,
        )
        return

    got = np.asarray(passage_ids)
    if got.shape != expected.shape:
        raise ValueError(
            "Passage-id alignment mismatch: dataset cache passages length does not match lexical cache. "
            f"subdataset={subdataset} got={got.shape} expected={expected.shape} (from {source}). "
            "Rebuild lexical caches and ensure dataset_id/subdataset match the cache."
        )

    if not np.array_equal(got.astype(str, copy=False), expected.astype(str, copy=False)):
        # Give a compact hint without dumping huge arrays.
        bad = int(np.sum(got.astype(str, copy=False)
                  != expected.astype(str, copy=False)))
        raise ValueError(
            "Passage-id alignment mismatch: dataset cache passage_ids do not match lexical cache ordering. "
            f"subdataset={subdataset} mismatched={bad}/{got.size} (checked against {source}). "
            "This reranker relies on lexical candidate indices mapping into the same passage row order. "
            "Rebuild caches from the same passages source."
        )


class BiEncoderCosineReranker:
    """Bi-encoder semantic reranker using batched embeddings + cosine similarity.

    - Encodes all queries in batches.
    - For each query, encodes only its lexical candidate passages (rerank prefix).
    - No corpus-wide ANN; strictly reranks within provided candidates.
    """

    def __init__(
        self,
        *,
        subdataset: str,
        passage_ids: np.ndarray,
        passage_texts: list[str],
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.subdataset = str(subdataset)
        self.passage_ids = np.asarray(passage_ids)
        self.passage_texts = list(passage_texts)
        self.model_name = str(model_name)
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        model_id_or_path: str
        try:
            # Prefer an existing local snapshot to avoid any network I/O.
            model_id_or_path = snapshot_download(
                repo_id=self.model_name,
                local_files_only=True,
            )
            logger.info("Using cached HF snapshot for model=%s",
                        self.model_name)
        except LocalEntryNotFoundError:
            model_id_or_path = self.model_name

        # Keep around so we can re-init the model on another device if needed.
        self._model_id_or_path = str(model_id_or_path)

        SentenceTransformer = _load_sentence_transformer()
        self.model = SentenceTransformer(self._model_id_or_path, device=self.device)
        # SentenceTransformer truncation behavior is controlled by max_seq_length.
        self.model.max_seq_length = int(self.max_length)

        if len(self.passage_texts) != int(self.passage_ids.shape[0]):
            raise ValueError(
                "passage_texts length must match passage_ids length, "
                f"got texts={len(self.passage_texts)} ids={self.passage_ids.shape[0]}"
            )

    def _switch_device(self, device: str) -> None:
        """Switch underlying SentenceTransformer device (best-effort).

        We re-initialize the SentenceTransformer to ensure its internal target
        device is consistent. This is more robust than trying to poke private
        attributes.
        """

        device_s = str(device)
        if device_s == self.device:
            return

        SentenceTransformer = _load_sentence_transformer()
        self.model = SentenceTransformer(self._model_id_or_path, device=device_s)
        self.model.max_seq_length = int(self.max_length)
        self.device = device_s

    @staticmethod
    def _is_no_kernel_image_cuda_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return (
            "no kernel image is available" in msg
            or "cudaerrornokernelimagefordevice" in msg
            or "no kernel image" in msg
        )

    @classmethod
    def from_dataset_cache(
        cls,
        *,
        dataset_id: str,
        subdataset: str,
        model_name: str,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> "BiEncoderCosineReranker":
        passages = _load_passages_from_dataset_cache(
            dataset_id=str(dataset_id), subdataset=str(subdataset)
        )

        # Fail fast if lexical caches exist and disagree on ordering.
        _verify_alignment_with_lexical_caches(
            subdataset=str(subdataset), passage_ids=passages.ids
        )

        resolved_device = _default_device(device)
        logger.info(
            "Loaded %s passages for subdataset=%s; initializing bi-encoder model=%s device=%s",
            int(passages.ids.shape[0]),
            subdataset,
            model_name,
            resolved_device,
        )

        return cls(
            subdataset=str(subdataset),
            passage_ids=passages.ids,
            passage_texts=passages.texts,
            model_name=str(model_name),
            device=resolved_device,
            batch_size=int(batch_size),
            max_length=int(max_length),
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
            raise ValueError("top_n must be <= rerank_k")

        torch = _torch()

        logger.info(
            "Starting bi-encoder rerank: nq=%s K=%s rerank_k=%s top_n=%s alpha=%s device=%s batch_size=%s",
            nq,
            K,
            rk,
            top_n,
            alpha,
            self.device,
            self.batch_size,
        )

        tq0 = time.monotonic()
        logger.info("Encoding %s queries...", nq)
        q_emb = None
        try:
            with torch.no_grad():
                q_emb = self.model.encode(
                    [str(t) for t in query_texts],
                    batch_size=int(self.batch_size),
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                )
        except Exception as e:  # noqa: BLE001
            # Common failure mode: GPU exists, but torch CUDA build doesn't support its compute capability.
            if str(self.device).startswith("cuda") and self._is_no_kernel_image_cuda_error(e):
                logger.warning(
                    "CUDA kernel image not available for this GPU with current PyTorch build. "
                    "Falling back to CPU. Original error: %s",
                    e,
                )
                self._switch_device("cpu")
                with torch.no_grad():
                    q_emb = self.model.encode(
                        [str(t) for t in query_texts],
                        batch_size=int(self.batch_size),
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
            else:
                raise
        assert q_emb is not None
        logger.info("Encoded queries in %.1fs", time.monotonic() - tq0)

        out_idx = np.empty((nq, top_n), dtype=np.int32)
        out_scores = np.empty((nq, top_n), dtype=np.float32)
        out_ids = np.empty(
            (nq, top_n), dtype=np.asarray(self.passage_ids).dtype)

        last_log_t = time.monotonic()
        for i in range(nq):
            idx_row = cand_idx[i][:rk]
            lex_row = cand_lex[i][:rk]

            cand_texts = [self.passage_texts[int(j)] for j in idx_row.tolist()]
            try:
                with torch.no_grad():
                    p_emb = self.model.encode(
                        cand_texts,
                        batch_size=int(self.batch_size),
                        convert_to_tensor=True,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
            except Exception as e:  # noqa: BLE001
                if str(self.device).startswith("cuda") and self._is_no_kernel_image_cuda_error(e):
                    logger.warning(
                        "CUDA kernel image not available during passage encoding; falling back to CPU. "
                        "Original error: %s",
                        e,
                    )
                    self._switch_device("cpu")
                    # Ensure query embeddings are on CPU too.
                    q_emb = q_emb.detach().to("cpu")
                    with torch.no_grad():
                        p_emb = self.model.encode(
                            cand_texts,
                            batch_size=int(self.batch_size),
                            convert_to_tensor=True,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                        )
                else:
                    raise

            # Cosine similarity for normalized embeddings is dot product.
            sem = (p_emb @ q_emb[i]
                   ).detach().to("cpu").to(torch.float32).numpy()
            sem_scores = np.asarray(sem, dtype=np.float32)

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

            now = time.monotonic()
            if now - last_log_t >= 5.0 or (i + 1) == nq:
                pct = 100.0 * float(i + 1) / float(nq)
                logger.info("Rerank progress: %s/%s (%.1f%%)", i + 1, nq, pct)
                last_log_t = now

        return RerankResult(indices=out_idx, ids=out_ids, scores=out_scores)
