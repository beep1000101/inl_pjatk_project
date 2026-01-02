# `src/config/` – Configuration and canonical paths

This module exists to make the project **portable and reproducible**.

In PolEval Task 3 you have multiple datasets / subdatasets (trivia, legal, customer support) and a lot of generated artifacts (vectorizers, sparse matrices, submissions). If paths and dataset IDs are scattered across scripts, the project quickly becomes brittle.

`src/config/` centralizes two things:

1) **Where things live** (on disk)
2) **Which dataset source is being used** (the Hugging Face dataset id)

## What’s inside

- `paths.py`
  - Single source of truth for:
    - repository root
    - `.cache/` structure
    - PolEval file conventions (`passages.jl`, `questions-*.jl`, `pairs-*.tsv`)
  - The most important functions:
    - `poleval2022_passages_path(dataset_id, subdataset)`
    - `poleval2022_questions_path(dataset_id, subdataset, split)`
    - `poleval2022_pairs_path(dataset_id, subdataset, split)`

- `config.toml`
  - Human-editable configuration. In practice the key piece is the dataset id used by [src/data_source/](../data_source/README.md).

- `config_parser.py`
  - Reads `config.toml` and exposes typed accessors such as `get_data_source_dataset_id()`.

## Why this helps for the PolEval task

PolEval data is distributed as multiple directories with consistent filenames. By encoding those conventions in `paths.py`:

- evaluation scripts don’t need to “know” directory layouts;
- preprocessing and reranking modules can reliably find the same passage ordering;
- caching stays consistent across runs (critical for comparing runs fairly).

## Typical usage

Most modules import paths like:

- `from src.config.paths import CACHE_DIR, poleval2022_passages_path`

This keeps the rest of the code focused on retrieval logic rather than file plumbing.
