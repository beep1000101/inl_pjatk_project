# `src/data_source/` – Getting PolEval data into the local cache

This module exists because **everything else depends on the dataset being locally available**.

PolEval Task 3 provides:
- `passages.jl` (the corpus)
- `questions-*.jl` (questions for a split)
- `pairs-*.tsv` (labels, when available)

The repository is structured so that we always work from a local cache under `.cache/data/…`.

## Why a dedicated module?

For a final project, you want runs that are:
- easy to reproduce on another machine,
- independent of “did you manually download/unzip the dataset?”,
- consistent with the path layout expected by preprocessing and evaluation.

Using `huggingface_hub.snapshot_download` provides exactly that: it downloads the dataset repository as-is (directory structure preserved), without requiring `git-lfs`.

## What’s inside

- `get_data.py`
  - `snapshot_dataset(force_reload=False)`: downloads the dataset repo specified in `src/config/config.toml` into `.cache/data/<dataset-id>/…`.
  - `main()`: a small CLI entrypoint that downloads and prints a sanity check of expected subdirectories.

## How it helps solve the PolEval task

All downstream code assumes you can do:

- load passages via `poleval2022_passages_path(dataset_id, subdataset)`
- load questions via `poleval2022_questions_path(dataset_id, subdataset, split)`
- load labels (optional) via `poleval2022_pairs_path(dataset_id, subdataset, split)`

If the dataset isn’t in cache, nothing else can run.

## Typical usage

From repo root:

- `python -m src.data_source.get_data`

After this step, you should have:
- `.cache/data/piotr-rybak__poleval2022-passage-retrieval-dataset/wiki-trivia/passages.jl`
- `.cache/data/piotr-rybak__poleval2022-passage-retrieval-dataset/wiki-trivia/questions-test.jl`
- etc.
