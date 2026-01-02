from pathlib import Path
import json
import sys
import logging

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    # Preferred when running from repo root (most of this repo uses `src.*`).
    from src.config.paths import CACHE_DIR, poleval2022_passages_path
except ModuleNotFoundError:  # pragma: no cover
    # Fallback for environments that put `src/` directly on PYTHONPATH.
    from config.paths import CACHE_DIR, poleval2022_passages_path


logger = logging.getLogger(__name__)


POLEVAL2022_DATASETS = {"wiki-trivia", "allegro-faq", "legal-questions"}


def load_vectorizer(dataset_name: str) -> dict[str, object]:
    """Load cached TF-IDF artifacts for a dataset.

    Expects artifacts in:
      .cache/preprocessed_data/tf_idf_vectors/<dataset_name>/

    Returns a dict with keys: out_dir, vectorizer, matrix, passage_ids, meta.
    """
    if dataset_name not in POLEVAL2022_DATASETS:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Expected one of: wiki-trivia, allegro-faq, legal-questions"
        )

    out_dir = CACHE_DIR / "preprocessed_data" / "tf_idf_vectors" / dataset_name
    vectorizer_path = out_dir / "vectorizer.joblib"
    matrix_path = out_dir / "passages_tfidf.npz"
    ids_path = out_dir / "passage_ids.npy"
    meta_path = out_dir / "meta.json"

    missing = [
        p.name
        for p in (vectorizer_path, matrix_path, ids_path, meta_path)
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing cached artifacts for {dataset_name} in {out_dir}: {', '.join(missing)}. "
            f"Run: python -m src.preprocess.tf_idf_vectors {dataset_name}"
        )

    logger.info("Loading cached TF-IDF artifacts from: %s", out_dir)
    vectorizer = joblib.load(vectorizer_path)
    matrix = sparse.load_npz(matrix_path)
    passage_ids = np.load(ids_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "out_dir": out_dir,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "passage_ids": passage_ids,
        "meta": meta,
    }


def get_tf_idf_vectorizer(min_df=5,
                          max_df=0.9,
                          max_features=500_000,
                          dtype=np.float32):
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        dtype=dtype,
    )
    return vectorizer


def read_jsonl(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def drop_redirects(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df["text"].str.startswith(("REDIRECT", "PATRZ"), na=False)
    result = df[mask]
    return result


def join_title_and_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    result = title + " " + text
    return result


def preprocess_passages(passage_data: pd.DataFrame) -> pd.Series:
    corpus = passage_data.pipe(drop_redirects).pipe(join_title_and_text)
    return corpus


def load_passages(passages_path: Path, max_rows: int | None = None) -> pd.DataFrame:
    data = read_jsonl(passages_path, max_rows=max_rows)
    return data


def create_wiki_trivia_tf_idf(
    dataset_id: str = "piotr-rybak__poleval2022-passage-retrieval-dataset",
    *,
    min_df=5,
    max_df=0.9,
    max_features=500_000,
    dtype=np.float32,
    max_rows: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Create and cache TF-IDF vectors for the wiki-trivia passages.

    Saves artifacts to:
      .cache/preprocessed_data/tf_idf_vectors/wiki-trivia/
    """
    subdataset = "wiki-trivia"
    passages_path = poleval2022_passages_path(dataset_id, subdataset)

    out_dir = CACHE_DIR / "preprocessed_data" / "tf_idf_vectors" / subdataset
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = out_dir / "vectorizer.joblib"
    matrix_path = out_dir / "passages_tfidf.npz"
    ids_path = out_dir / "passage_ids.npy"
    meta_path = out_dir / "meta.json"

    if (
        not force
        and vectorizer_path.is_file()
        and matrix_path.is_file()
        and ids_path.is_file()
        and meta_path.is_file()
    ):
        logger.info("Cache hit: %s", out_dir)
        return {
            "out_dir": out_dir,
            "vectorizer": vectorizer_path,
            "matrix": matrix_path,
            "passage_ids": ids_path,
            "meta": meta_path,
        }

    logger.info("Loading passages: %s", passages_path)
    passages_df = load_passages(passages_path, max_rows=max_rows)
    logger.info("Preprocessing passages")
    corpus = preprocess_passages(passages_df)
    passages_df = drop_redirects(passages_df).reset_index(drop=True)

    logger.info(
        "Fitting TF-IDF (min_df=%s, max_df=%s, max_features=%s)",
        min_df,
        max_df,
        max_features,
    )
    vectorizer = get_tf_idf_vectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        dtype=dtype,
    )
    X = vectorizer.fit_transform(corpus)

    passage_ids = np.asarray(passages_df["id"].astype(str).tolist(), dtype=str)

    logger.info("Saving artifacts: %s", out_dir)
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(matrix_path, X)
    np.save(ids_path, passage_ids, allow_pickle=False)
    meta_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "subdataset": subdataset,
                "passages_path": str(passages_path),
                "min_df": min_df,
                "max_df": max_df,
                "max_features": max_features,
                "dtype": str(dtype),
                "max_rows": max_rows,
                "n_passages": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "out_dir": out_dir,
        "vectorizer": vectorizer_path,
        "matrix": matrix_path,
        "passage_ids": ids_path,
        "meta": meta_path,
    }


PREPROCESSING_REGISTRY = {
    'wiki-trivia': create_wiki_trivia_tf_idf,
    'allegro-faq': NotImplementedError,
    'legal-questions': NotImplementedError
}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")
    if len(sys.argv) != 2:
        logger.error(
            "Usage: python -m src.preprocess.tf_idf_vectors <dataset_name> (supported: wiki-trivia)"
        )
        raise SystemExit(2)

    dataset_name = sys.argv[1]

    save_vectorizer_function = PREPROCESSING_REGISTRY.get(dataset_name)
    if save_vectorizer_function is None:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Expected one of: wiki-trivia, allegro-faq, legal-questions"
        )

    if save_vectorizer_function is NotImplementedError:
        raise NotImplementedError(
            f"TF-IDF caching not implemented for: {dataset_name}")

    paths = save_vectorizer_function()
    logger.info("TF-IDF (%s) cached in: %s", dataset_name, paths["out_dir"])


if __name__ == "__main__":
    main()
