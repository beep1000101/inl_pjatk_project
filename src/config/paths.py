from pathlib import Path


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
