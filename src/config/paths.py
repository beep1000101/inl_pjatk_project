from pathlib import Path


# Poleval 2022 dataset layout helpers
# Repo: piotr-rybak/poleval2022-passage-retrieval-dataset
# Subdirectories contain:
# - passages.jl
# - questions-<split>.jl
# - pairs-<split>.tsv


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

CONFIG_DIR = SRC_DIR / "config"
CONFIG_TOML_PATH = CONFIG_DIR / "config.toml"

CACHE_DIR = REPO_ROOT / ".cache"
CACHE_DATA_DIR = CACHE_DIR / "data"


def dataset_dirname(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def dataset_cache_dir(dataset_id: str) -> Path:
    return CACHE_DATA_DIR / dataset_dirname(dataset_id)


def dataset_dict_json_path(dataset_id: str) -> Path:
    return dataset_cache_dir(dataset_id) / "dataset_dict.json"


def dataset_split_dir(dataset_id: str, split: str) -> Path:
    return dataset_cache_dir(dataset_id) / split


POLEVAL2022_SUBDATASETS = (
    "wiki-trivia",
    "legal-questions",
    "allegro-faq",
)


def poleval2022_subdataset_dir(dataset_id: str, subdataset: str) -> Path:
    return dataset_cache_dir(dataset_id) / subdataset


def poleval2022_passages_path(dataset_id: str, subdataset: str) -> Path:
    return poleval2022_subdataset_dir(dataset_id, subdataset) / "passages.jl"


def poleval2022_questions_path(dataset_id: str, subdataset: str, split: str) -> Path:
    return poleval2022_subdataset_dir(dataset_id, subdataset) / f"questions-{split}.jl"


def poleval2022_pairs_path(dataset_id: str, subdataset: str, split: str) -> Path:
    return poleval2022_subdataset_dir(dataset_id, subdataset) / f"pairs-{split}.tsv"
