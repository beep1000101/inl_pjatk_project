from __future__ import annotations

from pathlib import Path

import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    st.set_page_config(
        page_title="PolEval 2022 Passage Retrieval — wiki-trivia",
        layout="wide",
    )

    st.title("PolEval 2022 Task 3: Passage Retrieval — wiki-trivia")

    st.markdown(
        """
This Streamlit app presents the existing experiment results in this repository, scoped strictly to:

- `dataset_id = piotr-rybak__poleval2022-passage-retrieval-dataset`
- `subdataset = wiki-trivia`

Use the pages in the left sidebar to navigate:

- Problem summary
- Lexical baselines (TF‑IDF, BM25)
- Lexical + semantic re-ranking (hybrid)
- Final summary

Notes:
- All numeric values shown in this app are loaded from `.cache/submissions/**/metrics.csv`.
- The app does not use `legal-questions` or `allegro-faq`.
"""
    )

    st.caption(f"Project root: {_project_root()}")


if __name__ == "__main__":
    main()
