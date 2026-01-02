from __future__ import annotations

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="PolEval 2022 — wyszukiwanie fragmentów (wiki-trivia)",
        layout="wide",
    )

    st.title("PolEval 2022 Zadanie 3: Wyszukiwanie fragmentów — wiki-trivia")

    st.markdown(
        """
Ta aplikacja Streamlit prezentuje wyniki eksperymentów (z wcześniej uruchomionych przebiegów), w zakresie:

- `dataset_id = piotr-rybak__poleval2022-passage-retrieval-dataset`
- `subdataset = wiki-trivia`

Do nawigacji użyj stron w lewym panelu:

- Podsumowanie problemu
- Bazy leksykalne (TF‑IDF, BM25)
- Reranking leksykalny + semantyczny (hybryda)
- Podsumowanie końcowe

Uwagi:
- Wszystkie wartości liczbowe w aplikacji są wczytywane z zapisanych metryk (np. `app/data/submissions/**/metrics.csv`).
- Prezentacja dotyczy wyłącznie `wiki-trivia`.
"""
    )


if __name__ == "__main__":
    main()
