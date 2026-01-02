from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.config.paths import CACHE_DIR
from src.hybrid.lexical import (
    Bm25LexicalRetriever,
    LexicalRetriever,
    TfidfLexicalRetriever,
)
from src.hybrid.semantic import FastTextCosineReranker
from src.models.fasttext import load_fasttext_model
from src.preprocess.fasttext_vectors import load_fasttext_vectors, tokenize


LexicalMethod = Literal["tfidf", "bm25"]


def _default_fasttext_model_path(lang: str = "pl", dim: int = 300) -> Path:
    return CACHE_DIR / "fasttext" / f"cc.{lang}.{dim}.bin"


@dataclass
class HybridPipeline:
    lexical: LexicalRetriever
    reranker: FastTextCosineReranker
    passage_vectors: np.ndarray
    passage_ids: np.ndarray

    def retrieve(
        self,
        query_texts: list[str],
        *,
        top_k_candidates: int = 500,
        top_n: int = 10,
        lexical: LexicalMethod | None = None,
        chunk_size: int = 10_000,
        progress: Any | None = None,
        **lexical_kwargs: Any,
    ) -> np.ndarray:
        """Hybrid retrieval: lexical candidate generation -> semantic reranking.

        Returns (n_queries, top_n) passage id strings.
        """

        # lexical argument is accepted for API ergonomics; the pipeline is already configured.
        _ = lexical

        candidates = self.lexical.retrieve(
            query_texts,
            top_k_candidates=int(top_k_candidates),
            chunk_size=int(chunk_size),
            progress=progress,
            **lexical_kwargs,
        )

        reranked = self.reranker.rerank(
            query_texts=[str(t) for t in query_texts],
            candidate_indices=candidates.indices,
            candidate_lexical_scores=candidates.scores,
            passage_vectors=self.passage_vectors,
            passage_ids=self.passage_ids,
            top_n=int(top_n),
        )

        return reranked.ids


def build_hybrid_pipeline(
    *,
    subdataset: str,
    lexical: LexicalMethod = "bm25",
    fasttext_model_path: Path | None = None,
) -> HybridPipeline:
    """Convenience constructor that loads cached artifacts.

    This function does NOT download models or build caches.
    It expects all required artifacts to already exist under `.cache/preprocessed_data/`.
    """

    if lexical == "tfidf":
        lexical_retriever: LexicalRetriever = TfidfLexicalRetriever.from_cache(
            subdataset)
    elif lexical == "bm25":
        lexical_retriever = Bm25LexicalRetriever.from_cache(subdataset)
    else:
        raise ValueError(f"Unknown lexical method: {lexical}")

    ft = load_fasttext_vectors(subdataset)
    passage_vectors = np.asarray(ft["vectors"])
    ft_ids = np.asarray(ft["passage_ids"]).astype(str, copy=False)

    lex_ids = np.asarray(lexical_retriever.passage_ids).astype(str, copy=False)
    if lex_ids.shape != ft_ids.shape or not np.array_equal(lex_ids, ft_ids):
        raise ValueError(
            "Passage id alignment mismatch between lexical artifacts and fastText vectors. "
            "Rebuild caches using the same preprocessing pipeline so passage order matches."
        )

    model_path = fasttext_model_path or _default_fasttext_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"fastText model not found at: {model_path}. "
            "Provide --fasttext-model-path pointing to an existing .bin (no auto-download in hybrid scripts)."
        )

    model = load_fasttext_model(model_path=model_path)
    reranker = FastTextCosineReranker(model=model, tokenize=tokenize)

    return HybridPipeline(
        lexical=lexical_retriever,
        reranker=reranker,
        passage_vectors=passage_vectors,
        passage_ids=ft_ids,
    )
