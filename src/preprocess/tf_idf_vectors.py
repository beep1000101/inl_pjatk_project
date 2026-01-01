from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.paths import CACHE_DIR, poleval2022_passages_path


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


def drop_redirects(df):
    mask = ~df["text"].str.startswith(("REDIRECT", "PATRZ"), na=False)
    result = df[mask]
    return result


def join_title_and_text(df):
    result = (df["title"].fillna("") + " " + df["text"].fillna(""))
    return result


def preprocess_passages(passage_data):
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
        return {
            "out_dir": out_dir,
            "vectorizer": vectorizer_path,
            "matrix": matrix_path,
            "passage_ids": ids_path,
            "meta": meta_path,
        }

    passages_df = load_passages(passages_path, max_rows=max_rows)
    passages_df = drop_redirects(passages_df).reset_index(drop=True)
    corpus = passages_df.pipe(join_title_and_text)

    vectorizer = get_tf_idf_vectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        dtype=dtype,
    )
    X = vectorizer.fit_transform(corpus)

    passage_ids = np.asarray(passages_df["id"].astype(str).tolist(), dtype=str)

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


def main():
    ...
