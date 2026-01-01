from __future__ import annotations

from dataclasses import dataclass
import gzip
import logging
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

from src.config.paths import CACHE_DIR


logger = logging.getLogger(__name__)


FASTTEXT_CACHE_DIR = CACHE_DIR / "fasttext"


@dataclass(frozen=True)
class FastTextModelSpec:
    """Spec for a pretrained fastText model hosted by Facebook AI (fastText)."""

    lang: str = "pl"
    dim: int = 300

    @property
    def filename_bin(self) -> str:
        return f"cc.{self.lang}.{self.dim}.bin"

    @property
    def filename_gz(self) -> str:
        return f"{self.filename_bin}.gz"

    @property
    def url(self) -> str:
        return (
            f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{self.lang}.{self.dim}.bin.gz"
        )


def ensure_fasttext_model(spec: FastTextModelSpec = FastTextModelSpec(), *, force: bool = False) -> Path:
    """Ensure a pretrained fastText `.bin` model is available in `.cache/fasttext/`.

    Implementation (as requested):
    - download `.bin.gz` to a temporary file
    - decompress once into cached `.bin`
    - delete the `.gz`

    Returns:
        Path to the cached `.bin` file.
    """
    FASTTEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    bin_path = FASTTEXT_CACHE_DIR / spec.filename_bin
    if bin_path.is_file() and not force:
        return bin_path

    # Clean up stale partials if present
    (FASTTEXT_CACHE_DIR / spec.filename_gz).unlink(missing_ok=True)

    tmp_gz = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".bin.gz",
        prefix="fasttext_download_",
        dir=str(FASTTEXT_CACHE_DIR),
        delete=False,
    )
    tmp_gz_path = Path(tmp_gz.name)
    tmp_gz.close()

    tmp_bin = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".bin",
        prefix="fasttext_decompressed_",
        dir=str(FASTTEXT_CACHE_DIR),
        delete=False,
    )
    tmp_bin_path = Path(tmp_bin.name)
    tmp_bin.close()

    try:
        logger.info("Downloading fastText model: %s", spec.url)
        logger.info("Temporary target: %s", tmp_gz_path)
        urllib.request.urlretrieve(spec.url, tmp_gz_path)  # noqa: S310

        logger.info("Decompressing %s -> %s", tmp_gz_path, tmp_bin_path)
        with gzip.open(tmp_gz_path, "rb") as f_in, tmp_bin_path.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        tmp_gz_path.unlink(missing_ok=True)

        # Atomic replace to avoid leaving a half-written bin
        tmp_bin_path.replace(bin_path)
        logger.info("Cached fastText model: %s", bin_path)
        return bin_path
    except Exception:
        tmp_gz_path.unlink(missing_ok=True)
        tmp_bin_path.unlink(missing_ok=True)
        raise


def load_fasttext_model(
        *,
        model_path: Path | None = None,
        spec: FastTextModelSpec = FastTextModelSpec(),
        force_download: bool = False,
):
    """Load a fastText model.

    If `model_path` is None, downloads a default pretrained model into cache.
    """
    try:
        import fasttext  # type: ignore
    except ImportError as e:
        raise ImportError(
            "fasttext Python package is not installed. Install one of:\n"
            "  - pip install fasttext\n"
            "  - pip install fasttext-wheel"
        ) from e

    if model_path is not None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"fastText model file not found: {model_path}")
        bin_path = model_path
    else:
        bin_path = ensure_fasttext_model(spec, force=force_download)

    logger.info("Loading fastText model from: %s", bin_path)
    return fasttext.load_model(str(bin_path))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    # Minimal CLI: python -m src.models.fasttext [lang]
    lang = sys.argv[1] if len(sys.argv) >= 2 else "pl"
    spec = FastTextModelSpec(lang=lang, dim=300)
    path = ensure_fasttext_model(spec, force=False)
    print(path)


if __name__ == "__main__":
    main()
