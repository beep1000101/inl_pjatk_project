from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd


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


def load_passages(passage_path):
    data = read_jsonl(passages_path)
    return data
