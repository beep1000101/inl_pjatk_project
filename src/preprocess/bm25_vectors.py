from __future__ import annotations

from pathlib import Path
import argparse
import json
import logging
import re

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

try:
    from src.config.paths import CACHE_DIR, poleval2022_passages_path
except ModuleNotFoundError:  # pragma: no cover
    from config.paths import CACHE_DIR, poleval2022_passages_path


logger = logging.getLogger(__name__)

POLEVAL2022_DATASETS = {"wiki-trivia", "allegro-faq", "legal-questions"}

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    r"""Deterministic tokenization (lowercased \w+ tokens).

    Note: we do NOT pass this function into scikit-learn objects (to keep caches
    picklable across entrypoints). It is kept here only as an explicit reference
    for the intended tokenization scheme.
    """

    return _TOKEN_RE.findall(str(text).lower())


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
    return df[mask]


def join_title_and_text(df: pd.DataFrame) -> pd.Series:
    title = df["title"].fillna("").astype(str)
    text = df["text"].fillna("").astype(str)
    return title + " " + text


def preprocess_passages(passage_data: pd.DataFrame) -> pd.Series:
    return passage_data.pipe(drop_redirects).pipe(join_title_and_text)


def _bm25_idf(*, N: int, df: np.ndarray) -> np.ndarray:
    # Okapi BM25 idf (with +1 inside log to avoid negative idf for very frequent terms).
    df_f = df.astype(np.float32, copy=False)
    return np.log(((float(N) - df_f + 0.5) / (df_f + 0.5)) + 1.0).astype(np.float32)


def load_bm25(dataset_name: str) -> dict[str, object]:
    """Load cached BM25 artifacts.

    Expects artifacts in:
      .cache/preprocessed_data/bm25_vectors/<dataset_name>/

    Returns keys: out_dir, vectorizer, matrix, passage_ids, doc_len, idf, meta
    """

    if dataset_name not in POLEVAL2022_DATASETS:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Expected one of: wiki-trivia, allegro-faq, legal-questions"
        )

    out_dir = CACHE_DIR / "preprocessed_data" / "bm25_vectors" / dataset_name
    vectorizer_path = out_dir / "vectorizer.joblib"
    matrix_path = out_dir / "passages_tf.npz"
    ids_path = out_dir / "passage_ids.npy"
    doc_len_path = out_dir / "doc_len.npy"
    idf_path = out_dir / "idf.npy"
    meta_path = out_dir / "meta.json"

    missing = [
        p.name
        for p in (vectorizer_path, matrix_path, ids_path, doc_len_path, idf_path, meta_path)
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing cached artifacts for {dataset_name} in {out_dir}: {', '.join(missing)}. "
            f"Run: python -m src.preprocess.bm25_vectors {dataset_name}"
        )

    logger.info("Loading cached BM25 artifacts from: %s", out_dir)
    try:
        vectorizer = joblib.load(vectorizer_path)
    except AttributeError as e:
        # Backward-compat for older caches that pickled a custom tokenizer function.
        # Depending on how the cache was built, pickle may reference:
        # - __main__.tokenize (built via `python -m ...`)
        # - src.eval.hybrid.bm25_lsa_eval.tokenize (built via that module)
        # - other legacy modules
        # We can inject our local tokenize into the referenced module to unblock unpickling.
        import re as _re
        import sys as _sys
        import types as _types

        msg = str(e)
        m = _re.search(
            r"Can't get attribute 'tokenize' on <module '([^']+)'", msg)
        module_name = m.group(1) if m else None

        if "tokenize" not in msg:
            raise

        # Always try patching __main__ too, since that's a common legacy target.
        main_mod = _sys.modules.get("__main__")
        if main_mod is not None and not hasattr(main_mod, "tokenize"):
            setattr(main_mod, "tokenize", tokenize)

        if module_name is not None:
            mod = _sys.modules.get(module_name)
            if mod is None:
                mod = _types.ModuleType(module_name)
                _sys.modules[module_name] = mod
            if not hasattr(mod, "tokenize"):
                setattr(mod, "tokenize", tokenize)

        try:
            vectorizer = joblib.load(vectorizer_path)
        except AttributeError as e2:
            raise RuntimeError(
                f"Failed to load cached BM25 vectorizer from {vectorizer_path}. "
                "This usually means the cache was built with an older code version that "
                "pickled a custom tokenizer function. Rebuild the cache with: "
                f"python -m src.preprocess.bm25_vectors {dataset_name} --force"
            ) from e2
    matrix = sparse.load_npz(matrix_path).tocsr()
    passage_ids = np.load(ids_path, allow_pickle=False)
    doc_len = np.load(doc_len_path, allow_pickle=False)
    idf = np.load(idf_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {
        "out_dir": out_dir,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "passage_ids": passage_ids,
        "doc_len": doc_len,
        "idf": idf,
        "meta": meta,
    }


def create_wiki_trivia_bm25(
    dataset_id: str = "piotr-rybak__poleval2022-passage-retrieval-dataset",
    *,
    min_df: int = 5,
    max_df: float = 0.9,
    max_features: int = 500_000,
    max_rows: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    """Create and cache BM25 term-frequency artifacts for wiki-trivia passages."""

    subdataset = "wiki-trivia"
    passages_path = poleval2022_passages_path(dataset_id, subdataset)

    out_dir = CACHE_DIR / "preprocessed_data" / "bm25_vectors" / subdataset
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = out_dir / "vectorizer.joblib"
    matrix_path = out_dir / "passages_tf.npz"
    ids_path = out_dir / "passage_ids.npy"
    doc_len_path = out_dir / "doc_len.npy"
    idf_path = out_dir / "idf.npy"
    meta_path = out_dir / "meta.json"

    if (
        not force
        and vectorizer_path.is_file()
        and matrix_path.is_file()
        and ids_path.is_file()
        and doc_len_path.is_file()
        and idf_path.is_file()
        and meta_path.is_file()
    ):
        logger.info("Cache hit: %s", out_dir)
        return {
            "out_dir": out_dir,
            "vectorizer": vectorizer_path,
            "matrix": matrix_path,
            "passage_ids": ids_path,
            "doc_len": doc_len_path,
            "idf": idf_path,
            "meta": meta_path,
        }

    logger.info("Loading passages: %s", passages_path)
    passages_df = read_jsonl(passages_path, max_rows=max_rows)
    corpus = preprocess_passages(passages_df)
    passages_df = drop_redirects(passages_df).reset_index(drop=True)

    logger.info(
        "Fitting CountVectorizer (min_df=%s, max_df=%s, max_features=%s)",
        min_df,
        max_df,
        max_features,
    )
    # Keep this picklable and deterministic:
    # - token_pattern matches \w+ like the other baselines
    # - lowercase matches our intended tokenization
    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        token_pattern=r"(?u)\\b\\w+\\b",
        lowercase=True,
        dtype=np.int32,
    )

    X = vectorizer.fit_transform(corpus.astype(str).tolist()).tocsr()
    passage_ids = np.asarray(passages_df["id"].astype(str).tolist(), dtype=str)

    # doc length and idf
    doc_len = np.asarray(X.sum(axis=1)).ravel().astype(np.int32)
    df = X.getnnz(axis=0).astype(np.int32)
    idf = _bm25_idf(N=int(X.shape[0]), df=df)

    meta = {
        "dataset_id": dataset_id,
        "subdataset": subdataset,
        "passages_path": str(passages_path),
        "min_df": int(min_df),
        "max_df": float(max_df),
        "max_features": int(max_features),
        "max_rows": max_rows,
        "tokenizer": "countvectorizer(token_pattern=\\b\\w+\\b, lowercase=True)",
        "n_passages": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "avgdl": float(doc_len.mean()) if doc_len.size else 0.0,
        "idf": "okapi_log((N-df+0.5)/(df+0.5)+1)",
    }

    logger.info("Saving artifacts: %s", out_dir)
    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(matrix_path, X)
    np.save(ids_path, passage_ids, allow_pickle=False)
    np.save(doc_len_path, doc_len, allow_pickle=False)
    np.save(idf_path, idf, allow_pickle=False)
    meta_path.write_text(json.dumps(
        meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "out_dir": out_dir,
        "vectorizer": vectorizer_path,
        "matrix": matrix_path,
        "passage_ids": ids_path,
        "doc_len": doc_len_path,
        "idf": idf_path,
        "meta": meta_path,
    }


PREPROCESSING_REGISTRY = {
    "wiki-trivia": create_wiki_trivia_bm25,
    "allegro-faq": NotImplementedError,
    "legal-questions": NotImplementedError,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser(
        description="Cache BM25 (Okapi) artifacts (CountVectorizer TF matrix + idf + doc lengths)."
    )
    parser.add_argument(
        "dataset_name",
        help="Dataset name (supported: wiki-trivia)",
    )
    parser.add_argument("--force", action="store_true",
                        help="Rebuild cache even if present.")
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-df", type=float, default=0.9)
    parser.add_argument("--max-features", type=int, default=500_000)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    fn = PREPROCESSING_REGISTRY.get(dataset_name)
    if fn is None:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Expected one of: wiki-trivia, allegro-faq, legal-questions"
        )
    if fn is NotImplementedError:
        raise NotImplementedError(
            f"BM25 caching not implemented for: {dataset_name}")

    if dataset_name != "wiki-trivia":
        # Keep interface open for future datasets but explicit for now.
        paths = fn()  # type: ignore[misc]
    else:
        paths = create_wiki_trivia_bm25(
            min_df=int(args.min_df),
            max_df=float(args.max_df),
            max_features=int(args.max_features),
            max_rows=args.max_rows,
            force=bool(args.force),
        )
    logger.info("BM25 (%s) cached in: %s", dataset_name, paths["out_dir"])


if __name__ == "__main__":
    main()
