from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class RerankResult:
    indices: np.ndarray  # (nq, top_n)
    ids: np.ndarray  # (nq, top_n) dtype=str
    scores: np.ndarray  # (nq, top_n) float32


def minmax01(x: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo = float(np.min(x))
    hi = float(np.max(x))
    denom = max(hi - lo, float(eps))
    return (x - lo) / denom


class SemanticReranker(Protocol):
    def rerank(
        self,
        *,
        query_texts: list[str],
        candidate_indices: np.ndarray,  # (nq, K)
        candidate_lexical_scores: np.ndarray,  # (nq, K)
        top_n: int = 10,
        rerank_k: int | None = None,
        alpha: float | None = None,
    ) -> RerankResult: ...
