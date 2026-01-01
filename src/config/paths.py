from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"

CONFIG_DIR = SRC_DIR / "config"
CONFIG_TOML_PATH = CONFIG_DIR / "config.toml"

CACHE_DIR = REPO_ROOT / ".cache"
CACHE_DATA_DIR = CACHE_DIR / "data"
