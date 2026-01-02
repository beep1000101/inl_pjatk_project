# `src/hybrid/` – Hybrid retrieval: lexical candidates + semantic reranking

This module exists to explore a practical middle ground for PolEval Passage Retrieval:

- **Lexical retrieval** (BM25 / TF‑IDF) is strong and fast, especially in zero-shot settings.
- **Neural semantic retrieval** can capture meaning beyond exact token overlap, but is expensive at corpus scale.

A common production compromise is:

1) Retrieve a candidate set with a fast lexical method.
2) Rerank that candidate set with a more expensive semantic model.

That is exactly what this package implements.

## Why hybrid makes sense for PolEval Task 3

- The corpus can be millions of passages (Wikipedia), so “encode everything” is not feasible for reranking experiments.
- The competition is cross-domain, and for some domains you don’t have training pairs, so *fine-tuning dense retrievers* is not guaranteed to help.
- Hybrids let you keep lexical recall while testing whether semantic reranking improves the final top-10 ordering.

## What’s inside

### Lexical candidate generators

- `lexical.py`
  - Small wrappers that load cached lexical artifacts and expose a unified retrieval API.
  - Examples:
    - `Bm25LexicalRetriever.from_cache(subdataset)`
    - `TfidfLexicalRetriever.from_cache(subdataset)`

These retrievers return candidates by **indices** and **scores**, not only IDs. Indices matter because rerankers need to reference passage texts efficiently.

### Reranker protocol + utilities

- `reranker.py`
  - Defines the `RerankResult` shape and a `SemanticReranker` protocol.
  - Provides `minmax01()` normalization used for optional score fusion.

### Semantic rerankers

- `semantic_biencoder.py`
  - Implements `BiEncoderCosineReranker` using Sentence-Transformers.
  - Encodes queries + candidate passages and ranks by cosine similarity.
  - Includes practical robustness:
    - prefers local HF snapshots to avoid network I/O,
    - checks passage-id alignment with lexical caches,
    - can fall back to CPU if CUDA execution fails.

- `semantic_lsa.py`
  - Implements an LSA reranker (TruncatedSVD over TF‑IDF) as a lightweight semantic signal.
  - Useful when GPU is unavailable or when you want a fast reranker.

## The most important invariants

Hybrid reranking only works if the following is true:

- Lexical retrieval returns candidate indices that correspond to the **same passage ordering** used by the reranker’s `passage_texts`.

That is why `semantic_biencoder.py` performs an explicit alignment check against cached `passage_ids`.

## Where hybrid evaluation lives

The CLI runners that actually use this package are in:
- [src/eval/hybrid/](../eval/hybrid/README.md)

They combine a lexical retriever from this package with a reranker from this package, then evaluate using the shared evaluation helpers.
