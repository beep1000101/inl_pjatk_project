from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download (or verify cached) Sentence-Transformers bi-encoder model into the local "
            "Hugging Face cache (~/.cache/huggingface by default).\n\n"
            "This avoids network I/O during evaluation runs: once cached, the hybrid biencoder "
            "pipelines will load the local snapshot."
        )
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Hugging Face repo id for the Sentence-Transformers model.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional model revision (commit hash / tag).",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HF cache directory override (defaults to HF_HOME/huggingface).",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not download; fail if the model is not already cached.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    cache_dir: str | None = str(
        Path(args.cache_dir)) if args.cache_dir else None

    try:
        path = snapshot_download(
            repo_id=str(args.model_name),
            revision=str(args.revision) if args.revision else None,
            cache_dir=cache_dir,
            local_files_only=bool(args.local_files_only),
        )
    except LocalEntryNotFoundError as e:
        raise SystemExit(
            f"Model not found in local cache and --local-files-only was set: {args.model_name}"
        ) from e

    print(path)
    logger.info("Cached model snapshot: %s", path)


if __name__ == "__main__":
    main()
