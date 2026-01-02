# `src/calibration/` – Choosing hyperparameters by measuring Hits@k curves

This module exists because “pick a good `k`” is not just a preference in PolEval passage retrieval — it is an engineering decision with real consequences:

- Larger `k` (or larger candidate sets) improves recall and metrics *up to a point*,
- but also increases runtime and memory usage, especially for reranking.

For a final project write-up, it’s useful to justify why you chose values like:
- `k=10` (fixed by the task for submission)
- candidate set sizes like `top_k_candidates=200/500`
- reranking prefix sizes like `rerank_k=100/200`

## What’s inside

- `hits_at_k.py`
  - Runs retrieval once at a maximum K, then computes Hits@k efficiently for many smaller k values.
  - Writes tidy CSV outputs that are easy to plot or include in a report.
  - Supports both linear and log-spaced k schedules.

## Why this helps solve the PolEval task

PolEval evaluates the final ranking at top-10 (officially by nDCG@10), but while developing you often need to answer questions like:

- “Is my method mostly done improving by k=50?”
- “How big does my lexical candidate set need to be before reranking stops helping?”

Calibration makes these decisions **data-driven**, and it turns into report material:
- figures: Hits@k vs k
- tables: smallest k that reaches ≥90% of the max Hits@k

## Typical usage

From repo root (example; exact flags depend on your CLI):

- `python -m src.calibration.hits_at_k --subdataset wiki-trivia --split test --method bm25_okapi --max-k 200 --k-schedule log --k-points 30`

This will produce CSVs under `.cache/calibration/…` suitable for plotting.
