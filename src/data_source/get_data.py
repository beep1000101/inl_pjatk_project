from pathlib import Path

# Allow running this file directly (e.g. `python src/data_source/get_data.py`)
# by ensuring the repo root is on sys.path.
if __name__ == "__main__" and __package__ is None:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from huggingface_hub import snapshot_download

from src.config.config_parser import get_data_source_dataset_id
from src.config.paths import CACHE_DATA_DIR, dataset_cache_dir


def snapshot_dataset(force_reload: bool = False) -> Path:
    """
    Download the entire Hugging Face dataset repository into cache.

    This uses huggingface_hub.snapshot_download, which:
    - downloads ALL files (passages.jl, questions.jl, pairs.tsv, etc.)
    - preserves directory structure
    - works without git or git-lfs
    - is Docker- and CI-friendly

    Returns:
        Path to the local dataset root directory.
    """
    CACHE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset_id = get_data_source_dataset_id()
    target_dir = dataset_cache_dir(dataset_id)

    if target_dir.exists() and any(target_dir.iterdir()) and not force_reload:
        return target_dir

    snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    return target_dir


def main() -> None:
    path = snapshot_dataset(force_reload=False)
    print(f"Dataset snapshot available at: {path}")

    # Optional sanity check
    expected_dirs = [
        "train",
        "wiki-trivia",
        "legal-questions",
        "allegro-faq",
    ]

    for d in expected_dirs:
        p = path / d
        print(f"{d}: {'OK' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
