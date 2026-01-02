from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


ALLOWED_FIELDS = [
    "run_id",
    "method",
    "dataset_id",
    "subdataset",
    "questions_split",
    "pairs_split",
    "submission_only",
    "k",
    "n_questions",
    "n_labeled",
    "hits_at_k",
    "recall_at_k",
    "precision_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "top_k_candidates",
    "chunk_size",
    "rerank_k",
    "alpha",
    "bm25_k1",
    "bm25_b",
    "biencoder_model",
    "biencoder_device",
    "biencoder_batch_size",
    "biencoder_max_length",
]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=False)
def _load_toy_passages() -> pd.DataFrame:
    root = _project_root()
    path = root / "app" / "data" / "toy_passages.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_all_metrics() -> pd.DataFrame:
    root = _project_root()
    submissions_dir = root / ".cache" / "submissions"
    metric_files = sorted(submissions_dir.glob("**/metrics.csv"))

    frames: list[pd.DataFrame] = []
    for path in metric_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        df["_metrics_path"] = str(path)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    for col in ALLOWED_FIELDS:
        if col not in combined.columns:
            combined[col] = pd.NA

    combined = combined[[c for c in combined.columns if c in set(ALLOWED_FIELDS + ["_metrics_path"])]]
    return combined


def _wiki_trivia_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    return df[
        (df["dataset_id"] == "piotr-rybak__poleval2022-passage-retrieval-dataset")
        & (df["subdataset"] == "wiki-trivia")
    ].copy()


def page() -> None:
    st.title("Reranking leksykalny + semantyczny (hybryda)")
    st.caption("Główna metryka: ndcg_at_k (z metrics.csv)")

    st.markdown(
        """
### Dlaczego pipeline hybrydowy jest 2‑etapowy

To podejście wynika z praktycznego ograniczenia: korpus `wiki-trivia` ma miliony passage’y,
więc nie da się uruchamiać modelu semantycznego na całym korpusie dla każdego zapytania.

Stosuje się więc:

1) **Etap 1: generowanie kandydatów leksykalnie** (BM25 albo TF‑IDF), żeby dostać krótką listę
2) **Etap 2: reranking semantyczny** (bi-encoder) tylko wśród tych kandydatów

Reranking ma sens tylko wtedy, gdy trafny passage w ogóle znajdzie się w zbiorze kandydatów.
"""
    )

    df = _wiki_trivia_only(_load_all_metrics())
    if df.empty:
        st.error("Nie znaleziono metryk dla wiki-trivia w .cache/submissions/**/metrics.csv")
        return

    hybrid = df[df["method"].isin(["hybrid_bm25_biencoder", "hybrid_tfidf_biencoder"])].copy()
    if hybrid.empty:
        st.error("Nie znaleziono uruchomień hybrydowych (hybrid_*_biencoder).")
        return

    st.subheader("Uruchomienia (wiki-trivia)")
    show_cols = [
        "run_id",
        "method",
        "dataset_id",
        "subdataset",
        "questions_split",
        "pairs_split",
        "k",
        "n_questions",
        "n_labeled",
        "ndcg_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "hits_at_k",
        "top_k_candidates",
        "rerank_k",
        "chunk_size",
        "alpha",
        "bm25_k1",
        "bm25_b",
        "biencoder_batch_size",
        "biencoder_max_length",
        "submission_only",
    ]
    st.dataframe(hybrid[show_cols], width="stretch")

    st.subheader("Parametry pipeline")
    st.markdown(
        """
Pipeline hybrydowy ma 2 etapy:

- Etap 1 (leksykalny): pobierz `top_k_candidates` passage’y.
- Etap 2 (semantyczny): przestaw kolejność top `rerank_k` kandydatów bi-encoderem.

Pola w logach zawierają też „gałki” wpływające na czas działania:
- `chunk_size` dla przetwarzania korpusu
- batchowanie bi-encodera: `biencoder_batch_size`, `biencoder_max_length`
- urządzenie wykonania: `biencoder_device`

Uwaga o `alpha`:
- W dostępnych logach hybrydowych `alpha` jest puste w `metrics.csv`, co odpowiada `alpha=None`.
  W praktyce ranking jest wtedy napędzany wynikiem bi-encodera, a wynik leksykalny może działać jako tie‑breaker.
"""
    )

    st.subheader("Użyty bi-encoder (i dlaczego)")
    model_names = sorted(
        {
            str(x)
            for x in hybrid.get("biencoder_model", pd.Series(dtype=object)).dropna().unique().tolist()
            if str(x).strip()
        }
    )
    if model_names:
        if len(model_names) == 1:
            st.markdown(f"Model w tych uruchomieniach: `{model_names[0]}`")
        else:
            st.markdown("Modele w tych uruchomieniach:")
            st.markdown("\n".join([f"- `{m}`" for m in model_names]))
    else:
        st.markdown("Nazwa modelu nie występuje w cache’owanym `metrics.csv` dla tych uruchomień.")

    st.markdown(
        """
**Bi-encoder** koduje zapytanie oraz każdy kandydujący passage *niezależnie* do tej samej przestrzeni embeddingów.
Następnie punktuje kandydatów prostą funkcją podobieństwa (najczęściej cosine similarity / iloczyn skalarny).

Dlaczego to pasuje (intuicyjnie):
- **Wystarczająco szybki do rerankingu**: gdy etap leksykalny zawęzi listę do kilkuset kandydatów,
    bi-encoder potrafi efektywnie punktować shortlistę w batchach.
- **Bardziej ekspresywny niż czysta leksyka**: może łapać parafrazy/podobieństwo semantyczne, nawet gdy
    słowa kluczowe nie pokrywają się idealnie.

W logach są też parametry, które pomagają kontrolować koszt tego etapu (`biencoder_device`,
`biencoder_batch_size`, `biencoder_max_length`).
"""
    )

    st.subheader("Przykład zabawkowy: hybrydowy retrieval 2‑etapowy")
    st.caption("Przykładowe passage’e → leksykalne top‑j → reranking bi-encoderem → top‑k (indeksy do lookup)")

    toy = _load_toy_passages()
    if toy.empty:
        st.info("Nie znaleziono przykładowych passage’y w app/data/toy_passages.csv")
        return

    st.markdown("**Tabela przykładowych passage’y (10 wierszy)**")
    st.dataframe(toy, width="stretch")

    query = st.text_input("Zapytanie", value="Jaka jest stolica Francji?", key="toy_hybrid_query")

    j = 5
    top_k = 3

    st.markdown("↓ Etap 1: retrieval leksykalny (kandydaci top‑j)")
    st.markdown(
        r"""
Reprezentujemy pełen „korpus” 10 passage’y jako macierz leksykalną:

$$
M^{(lex)} \in \mathbb{R}^{10\times d_{lex}},\quad
M^{(lex)}=\begin{bmatrix}
m^{(lex)}_{0,0} & \cdots & m^{(lex)}_{0,d_{lex}-1}\\
\vdots & \ddots & \vdots\\
m^{(lex)}_{9,0} & \cdots & m^{(lex)}_{9,d_{lex}-1}
\end{bmatrix}
$$

i zapytanie jako wektor:

$$
v^{(lex)} \in \mathbb{R}^{d_{lex}},\quad
v^{(lex)}=[v^{(lex)}_0,\ldots,v^{(lex)}_{d_{lex}-1}]^\top
$$

Liczmy leksykalne podobieństwa cosinusowe:

$$
u^{(lex)} \in \mathbb{R}^{10},\quad
u^{(lex)}_i = \cos(M^{(lex)}_i, v^{(lex)}) = \frac{M^{(lex)}_i \cdot v^{(lex)}}{\lVert M^{(lex)}_i \rVert\,\lVert v^{(lex)} \rVert}
$$

(where $M^{(lex)}_i$ is the $i$-th row of $M^{(lex)}$).

Wybierz najlepszych $j$ kandydatów:

$$
\mathrm{indices}^{(lex)} = \operatorname{argmax}_{j}(u^{(lex)})
$$
"""
    )

    # Example candidate indices for demonstration.
    cand_indices = [2, 3, 8, 0, 9][:j]
    st.markdown(f"Przykład: **indeksy** (j={j}) = {cand_indices}")

    candidates = toy.iloc[cand_indices].copy()
    candidates.insert(0, "indeks", cand_indices)
    candidates.insert(1, "zapytanie", query)
    st.dataframe(candidates[["indeks", "zapytanie", "passage_id", "text"]], width="stretch")

    st.markdown("↓ Etap 2: reranking bi-encoderem (top‑k wśród kandydatów)")
    st.markdown(
        r"""
Teraz embedujemy tylko $j$ passage’y-kandydatów bi-encoderem (przestrzeń semantyczna):

$$
M^{(sem)} \in \mathbb{R}^{j\times d_{sem}},\quad
M^{(sem)}=\begin{bmatrix}
m^{(sem)}_{0,0} & \cdots & m^{(sem)}_{0,d_{sem}-1}\\
\vdots & \ddots & \vdots\\
m^{(sem)}_{j-1,0} & \cdots & m^{(sem)}_{j-1,d_{sem}-1}
\end{bmatrix}
$$

i zapytanie jako wektor:

$$
v^{(sem)} \in \mathbb{R}^{d_{sem}},\quad
v^{(sem)}=[v^{(sem)}_0,\ldots,v^{(sem)}_{d_{sem}-1}]^\top
$$

Liczmy semantyczne podobieństwa cosinusowe wśród kandydatów:

$$
u^{(sem)} \in \mathbb{R}^{j},\quad
u^{(sem)}_r = \cos(M^{(sem)}_r, v^{(sem)}) = \frac{M^{(sem)}_r \cdot v^{(sem)}}{\lVert M^{(sem)}_r \rVert\,\lVert v^{(sem)} \rVert}
$$

(gdzie $M^{(sem)}_r$ to $r$-ty wiersz $M^{(sem)}$, czyli $r$-ty kandydat z listy etapu 1).

Wybierz najlepszych $k$ kandydatów i zmapuj ich z powrotem na indeksy oryginalnych passage’y:

$$
\mathrm{indices}^{(sem)} = \operatorname{argmax}_{k}(u^{(sem)}),\quad
\mathrm{indices}^{(top\text{-}k)} = \mathrm{indices}^{(lex)}[\mathrm{indices}^{(sem)}]
$$
"""
    )

    # Example rerank rows within the candidate list.
    rerank_rows = [0, 2, 1][:top_k]
    topk_indices = [cand_indices[r] for r in rerank_rows]
    st.markdown(f"Przykład: **indeksy** (k={top_k}) = {topk_indices}")

    topk = toy.iloc[topk_indices].copy()
    topk.insert(0, "indeks", topk_indices)
    topk.insert(1, "zapytanie", query)
    st.dataframe(topk[["indeks", "zapytanie", "passage_id", "text"]], width="stretch")


page()
