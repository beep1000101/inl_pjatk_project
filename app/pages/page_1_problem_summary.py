from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def page() -> None:
    st.title("Podsumowanie problemu")

    st.markdown(
        r"""
## Opis zadania

To jest **PolEval 2022 Zadanie 3 (Passage Retrieval)**:

- Dla każdego pytania należy zwrócić **uporządkowaną listę 10 identyfikatorów passage’y** z dostarczonego korpusu.

Ta prezentacja jest **ograniczona wyłącznie do podzbioru `wiki-trivia`** (dane w cache):

- `.cache/data/piotr-rybak__poleval2022-passage-retrieval-dataset/wiki-trivia`

## Kontekst datasetu (wiki-trivia)

W `wiki-trivia` korpus to duża kolekcja **fragmentów Wikipedii** (podzielonych na kawałki przypominające akapity).
Zapytania to **pytania trivia / ogólna wiedza** (w stylu quizowym), a system ma zwrócić
**top‑10 identyfikatorów passage’y**, które najprawdopodobniej zawierają odpowiedź.

To uzasadnia dwie rodziny podejść:
- **Metody leksykalne (TF‑IDF, BM25)** są szybkie, interpretowalne i niewymagające treningu, ale bazują na nakładaniu
  się słów i nie modelują kolejności.
- **Reranking neuronowy/semantyczny** bywa bardziej „znaczeniowy”, ale jest zbyt drogi do uruchamiania na całym korpusie.
  Dlatego pipeline hybrydowy najpierw robi etap leksykalny (lejek), a potem przestawia kolejność tylko wśród kandydatów.
"""
    )

    st.subheader("Przykład zabawkowy (pytania + passage’e)")
    st.caption("Tylko ilustracja (to nie są prawdziwe wiersze z datasetu)")

    st.markdown("**Przykładowe pytania**")
    toy_questions = pd.DataFrame(
        [
        {"question_id": "q-001", "text": "Jaka jest stolica Francji?"},
        {"question_id": "q-002", "text": "Która planeta jest znana jako Czerwona Planeta?"},
        {"question_id": "q-003", "text": "Kto napisał 'Pana Tadeusza'?"},
        ]
    )
    st.dataframe(toy_questions, width="stretch")

    st.markdown("**Przykładowe passage’e (w stylu Wikipedii)**")
    toy_passages = pd.DataFrame(
        [
            {
                "passage_id": "p-10",
                "title": "Paris",
          "text": "Paryż jest stolicą i największym miastem Francji.",
            },
            {
                "passage_id": "p-22",
                "title": "Mars",
          "text": "Mars bywa nazywany Czerwoną Planetą ze względu na swój czerwonawy wygląd.",
            },
            {
                "passage_id": "p-31",
                "title": "Adam Mickiewicz",
          "text": "Adam Mickiewicz napisał poemat epicki 'Pan Tadeusz'.",
            },
            {
                "passage_id": "p-40",
                "title": "France",
          "text": "Francja to kraj w Europie Zachodniej, z wieloma miastami i regionami.",
            },
        ]
    )
    st.dataframe(toy_passages, width="stretch")

    st.markdown(
        """
  **Co zwraca system**

  Dla każdego pytania system zwraca uporządkowaną listę 10 identyfikatorów passage’y (top‑10). Na przykład:

  - dla `q-001`: `[p-10, p-40, ...]` (łącznie 10 ID)
"""
    )

    st.markdown(
        r"""
## Dane i ewaluacja

- **Pytania**: zapytania w języku naturalnym.
- **Korpus passage’y**: fragmenty Wikipedii.
- **Etykiety (labels)**: informacja, które passage’e są trafne dla danego pytania.

## Zaimplementowane rodziny metod

- **Bazy leksykalne**:
  - TF‑IDF + podobieństwo cosinusowe
  - BM25 (Okapi)

- **Hybrydy (kandydaci leksykalni + reranking semantyczny)**:
  - BM25 → reranker bi-encoder
  - TF‑IDF → reranker bi-encoder

Wszystkie metryki pokazywane w pozostałych częściach aplikacji pochodzą z logów eksperymentów zapisanych jako wiersze w:

- `.cache/submissions/<run_name>/metrics.csv`

(Na tej stronie nie ma metryk.)

## Metryki używane w aplikacji (wzory, bez notacji teoriomnogościowej)

Wszystkie metryki są liczone **dla progu $k$** (zwykle $k=10$), a następnie uśredniane.

Szczegóły implementacyjne:
- Trafność (relevance) jest **binarna** na podstawie `pairs-*.tsv`. Jeśli istnieje kolumna `score`, to tylko wiersze z `score > 0` są traktowane jako trafne.
- Metryki są liczone tylko dla **pytań z etykietami** i uśredniane po tych pytaniach.

Definicje:
- $N$: liczba pytań z etykietami.
- Dla każdego pytania $q$ model zwraca ranking top‑$k$ identyfikatorów passage’y $p_{q,1},\dots,p_{q,k}$.
- $y_{q,i}\in\{0,1\}$: czy passage na pozycji $i$ jest trafny dla pytania $q$.
- $n_{rel}(q)$: liczba trafnych passage’y dla pytania $q$ w etykietach.

Metryki raportowane w aplikacji:

- **Hits@k** (`hits_at_k`)
- **Recall@k** (`recall_at_k`)
- **Precision@k** (`precision_at_k`)
- **MRR@k** (`mrr_at_k`)
- **nDCG@k** (`ndcg_at_k`)

"""
    )

    st.subheader("Wzory metryk")

    st.markdown("**Hits@k** (`hits_at_k`) — uśrednione po pytaniach z etykietami")
    st.latex(
      r"\mathrm{Hits@k} = \frac{1}{N}\sum_{q=1}^{N} \mathbb{1}\left[\sum_{i=1}^{k} y_{q,i} > 0\right]"
    )

    st.markdown("**Recall@k** (`recall_at_k`) — uśrednione po pytaniach z etykietami")
    st.latex(
      r"\mathrm{Recall@k} = \frac{1}{N}\sum_{q=1}^{N} \frac{\sum_{i=1}^{k} y_{q,i}}{\max(1,n_{rel}(q))}"
    )

    st.markdown("**Precision@k** (`precision_at_k`) — uśrednione po pytaniach z etykietami")
    st.latex(
      r"\mathrm{Precision@k} = \frac{1}{N}\sum_{q=1}^{N} \frac{1}{k}\sum_{i=1}^{k} y_{q,i}"
    )

    st.markdown("**MRR@k** (`mrr_at_k`) — uśrednione po pytaniach z etykietami")
    st.latex(
      r"\mathrm{MRR@k} = \frac{1}{N}\sum_{q=1}^{N} \left(\max_{1\le i\le k} \frac{y_{q,i}}{i}\right)"
    )

    st.markdown("**nDCG@k** (`ndcg_at_k`) — uśrednione po pytaniach z etykietami")
    st.latex(r"\mathrm{DCG@k}(q) = \sum_{i=1}^{k} \frac{y_{q,i}}{\log_2(i+1)}")
    st.latex(r"\mathrm{IDCG@k}(q) = \sum_{i=1}^{\min(k,n_{rel}(q))} \frac{1}{\log_2(i+1)}")
    st.latex(
      r"\mathrm{nDCG@k} = \frac{1}{N}\sum_{q=1}^{N} \frac{\mathrm{DCG@k}(q)}{\mathrm{IDCG@k}(q)}"
    )

    root = _project_root()
    dataset_dir = (
        root
        / ".cache"
        / "data"
        / "piotr-rybak__poleval2022-passage-retrieval-dataset"
        / "wiki-trivia"
    )

    st.subheader("Lokalizacja datasetu (tylko wiki-trivia)")
    st.code(str(dataset_dir))


page()
