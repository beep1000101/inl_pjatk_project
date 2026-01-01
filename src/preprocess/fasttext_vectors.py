from pathlib import Path
import json
import logging
import re
import sys

import numpy as np
import pandas as pd

from src.config.paths import CACHE_DIR, poleval2022_passages_path
from src.models.fasttext import FastTextModelSpec, load_fasttext_model

logger = logging.getLogger(__name__)

POLEVAL2022_DATASETS = {"wiki-trivia", "allegro-faq", "legal-questions"}


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


_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def create_wiki_trivia_fasttext_vectors(
    dataset_id: str = "piotr-rybak__poleval2022-passage-retrieval-dataset",
    *,
    lang: str = "pl",
    max_rows: int | None = None,
    force: bool = False,
) -> dict[str, Path]:
    subdataset = "wiki-trivia"
    passages_path = poleval2022_passages_path(dataset_id, subdataset)

    out_dir = CACHE_DIR / "preprocessed_data" / "fasttext_vectors" / subdataset
    out_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = out_dir / "passages_ft.npy"
    ids_path = out_dir / "passage_ids.npy"
    meta_path = out_dir / "meta.json"

    if not force and vectors_path.is_file() and ids_path.is_file() and meta_path.is_file():
        logger.info("Cache hit: %s", out_dir)
        return {"out_dir": out_dir, "vectors": vectors_path, "passage_ids": ids_path, "meta": meta_path}

    passages_df = read_jsonl(passages_path, max_rows=max_rows)
    corpus = preprocess_passages(passages_df)
    passages_df = drop_redirects(passages_df).reset_index(drop=True)

    model = load_fasttext_model(spec=FastTextModelSpec(lang=lang, dim=300))
    dim = int(model.get_dimension())

    passage_ids = np.asarray(passages_df["id"].astype(str).tolist(), dtype=str)

    logger.info("Creating embeddings matrix: n=%s dim=%s", len(corpus), dim)
    vectors_mm = np.lib.format.open_memmap(
        vectors_path, mode="w+", dtype=np.float32, shape=(len(corpus), dim)
    )

    for i, text in enumerate(corpus.astype(str).tolist()):
        tokens = tokenize(text)
        if not tokens:
            vectors_mm[i] = 0.0
            continue

        v = np.zeros(dim, dtype=np.float32)
        for t in tokens:
            v += model.get_word_vector(t).astype(np.float32, copy=False)
        v /= float(len(tokens))
        vectors_mm[i] = v

        if (i + 1) % 50_000 == 0:
            logger.info("Embedded %s passages...", i + 1)

    np.save(ids_path, passage_ids, allow_pickle=False)
    meta_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "subdataset": subdataset,
                "passages_path": str(passages_path),
                "model": f"cc.{lang}.300.bin (cached)",
                "dim": dim,
                "max_rows": max_rows,
                "n_passages": int(len(corpus)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Saved fastText vectors in: %s", out_dir)
    return {"out_dir": out_dir, "vectors": vectors_path, "passage_ids": ids_path, "meta": meta_path}


def load_fasttext_vectors(dataset_name: str) -> dict[str, object]:
    if dataset_name not in POLEVAL2022_DATASETS:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Expected one of: wiki-trivia, allegro-faq, legal-questions"
        )

    out_dir = CACHE_DIR / "preprocessed_data" / "fasttext_vectors" / dataset_name
    vectors_path = out_dir / "passages_ft.npy"
    ids_path = out_dir / "passage_ids.npy"
    meta_path = out_dir / "meta.json"

    missing = [p.name for p in (
        vectors_path, ids_path, meta_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing cached artifacts for {dataset_name} in {out_dir}: {', '.join(missing)}. "
            f"Run: python -m src.preprocess.fasttext_vectors {dataset_name}"
        )

    vectors = np.load(vectors_path, mmap_mode="r")
    passage_ids = np.load(ids_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return {"out_dir": out_dir, "vectors": vectors, "passage_ids": passage_ids, "meta": meta}


PREPROCESSING_REGISTRY = {
    "wiki-trivia": create_wiki_trivia_fasttext_vectors,
    "allegro-faq": NotImplementedError,
    "legal-questions": NotImplementedError,
}


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s:%(name)s:%(message)s")

    if len(sys.argv) != 2:
        logger.error(
            "Usage: python -m src.preprocess.fasttext_vectors <dataset_name>"
        )
        raise SystemExit(2)

    dataset_name = sys.argv[1]
    save_vectors = PREPROCESSING_REGISTRY.get(dataset_name)
    if save_vectors is None:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. Expected one of: wiki-trivia, allegro-faq, legal-questions"
        )
    if save_vectors is NotImplementedError:
        raise NotImplementedError(
            f"fastText vectors not implemented for: {dataset_name}")

    paths = save_vectors()
    logger.info("fastText (%s) cached in: %s", dataset_name, paths["out_dir"])


if __name__ == "__main__":
    main()
