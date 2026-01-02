from __future__ import annotations

from pathlib import Path

import altair as alt
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
    # Keep a stable integer index for the "lookup indices" explanation.
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


@st.cache_data(show_spinner=False)
def _load_calibration_hits_points() -> pd.DataFrame:
    root = _project_root()
    path = (
        root
        / ".cache"
        / "calibration"
        / "wiki-trivia"
        / "test"
        / "hits_points_maxk200_log_p20.csv"
    )
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["_calibration_path"] = str(path)
    return df


def page() -> None:
    st.title("Bazy leksykalne")
    st.caption("Główna metryka: ndcg_at_k (z metrics.csv)")

    st.markdown(
        """
### Dlaczego zaczynamy od baz leksykalnych

TF‑IDF i BM25 to solidne „baseline’y” na start, bo:

- Nie wymagają treningu (**training‑free**) i łatwo je odtwarzać.
- Dobrze skalują się do dużych korpusów, gdy reprezentacje passage’y są **cache’owane**.
"""
    )

    st.subheader("Jak powstaje przestrzeń leksykalna")
    st.markdown(
        r"""
Artefakty leksykalne są liczone raz i cache’owane w
`.cache/preprocessed_data/<method>/<subdataset>/...`.
To przyspiesza ewaluację (bez ponownej wektoryzacji milionów passage’y) i utrzymuje stabilne
indeksowanie wierszy, dzięki czemu „top indeksy → lookup” jest powtarzalne.

Wspólne przetwarzanie passage’y (TF‑IDF i BM25):
- Drop redirect-like entries where `text` starts with `REDIRECT` or `PATRZ`.
- Build the text used for vectorization as `title + " " + text`.
"""
    )

    st.markdown(
    r"""
#### TF‑IDF (used by `tfidf_cosine`)

Implementacja:
- Uses `sklearn.feature_extraction.text.TfidfVectorizer` (defaults: `min_df=5`, `max_df=0.9`,
    `max_features=500_000`, `dtype=float32`).
- Fits on the passage corpus and produces a sparse TF‑IDF matrix $M \in \mathbb{R}^{N\times d}$.

Jak to działa (intuicyjnie):
- **TF** (term frequency) odpowiada na pytanie „czy ten passage często używa danego słowa?”.
- **IDF** (inverse document frequency) obniża wagę słów częstych „wszędzie” (stop‑word-ish)
    i podbija wagę słów rzadszych, czyli bardziej rozróżniających.
- Wynikiem jest przestrzeń wektorowa, gdzie zapytanie i passage są „blisko”, jeśli dzielą informatywne słowa.

Dlaczego to działa w tym zadaniu:
- Pytania często zawierają mocne sygnały leksykalne (nazwy, encje, kluczowe frazy).
- TF‑IDF jest training‑free, szybkie po zbudowaniu cache i daje dobry punkt odniesienia.

Artefakty w cache:
- `vectorizer.joblib` (wytrenowany vectorizer)
- `passages_tfidf.npz` (macierz rzadka)
- `passage_ids.npy` (indeks wiersza → passage_id)
- `meta.json` (parametry + rozmiary)
"""
    )

    st.markdown(
    r"""
#### BM25 (used by `bm25_okapi`)

Implementacja:
- Builds a sparse **term-frequency** matrix using `sklearn.feature_extraction.text.CountVectorizer`
    with `token_pattern=r"(?u)\b\w+\b"`, `lowercase=True`, `dtype=int32`.
- Computes document lengths `doc_len` and an Okapi-style idf vector (saved to disk).

Jak to działa (intuicyjnie):
- BM25 jest podobne do TF‑IDF, ale ma dwa ważne usprawnienia:
    - **Saturacja**: powtarzanie słowa pomaga, ale z malejącymi korzyściami.
    - **Normalizacja długości**: bardzo długie passage’e nie powinny wygrywać tylko dlatego, że mają więcej słów.

Dlaczego to działa w tym zadaniu:
- Passage’e Wikipedii mocno różnią się długością; normalizacja BM25 dobrze do tego pasuje.
- To nadal metoda czysto leksykalna (szybka, training‑free), więc dobrze nadaje się jako etap 1 do generowania
    kandydatów dla rerankerów.

Artefakty w cache:
- `vectorizer.joblib`, `passages_tf.npz`, `passage_ids.npy`
- `doc_len.npy`, `idf.npy`, `meta.json`
"""
    )

    df = _wiki_trivia_only(_load_all_metrics())
    if df.empty:
        st.error("Nie znaleziono metryk dla wiki-trivia w .cache/submissions/**/metrics.csv")
        return

    lexical = df[df["method"].isin(["tfidf_cosine", "bm25_okapi"])].copy()
    if lexical.empty:
        st.error("Nie znaleziono uruchomień baz leksykalnych (tfidf_cosine / bm25_okapi).")
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
        "bm25_k1",
        "bm25_b",
        "submission_only",
        "chunk_size",
    ]
    st.dataframe(lexical[show_cols], width="stretch")

    st.subheader("Przykład zabawkowy: retrieval leksykalny w przestrzeni wektorowej")
    st.caption("Minimalny przebieg: passage’e → (wymyślona) macierz M → wektor zapytania v → podobieństwo cosinusowe → top indeksy → lookup")

    toy = _load_toy_passages()
    if toy.empty:
        st.info("Nie znaleziono przykładowych passage’y w app/data/toy_passages.csv")
    else:
        st.markdown("**Tabela przykładowych passage’y (10 wierszy)**")
        st.dataframe(toy, width="stretch")

        query = st.text_input("Zapytanie", value="Jaka jest stolica Francji?")

        st.markdown("↓ Rzutuj passage’e do przestrzeni wektorowej")

        st.markdown(
            r"""
Reprezentujemy 10 passage’y jako macierz w pewnej przestrzeni wektorowej:

$$
M \in \mathbb{R}^{10\times d},\quad
M=\begin{bmatrix}
m_{0,0} & \cdots & m_{0,d-1}\\
\vdots & \ddots & \vdots\\
m_{9,0} & \cdots & m_{9,d-1}
\end{bmatrix}
$$

i zapytanie jako wektor:

$$
v \in \mathbb{R}^{d},\quad
v = [v_0,\ldots,v_{d-1}]^\top
$$
"""
        )

        st.markdown("↓ Policz podobieństwo cosinusowe")
        st.markdown(
            r"""
Wyznaczamy wektor wyników $u$ licząc podobieństwo cosinusowe między każdym wierszem $M_i$ a wektorem zapytania $v$:

$$
u \in \mathbb{R}^{10},\quad
u_i = \cos(M_i, v) = \frac{M_i \cdot v}{\lVert M_i \rVert\,\lVert v \rVert}
$$

(gdzie $M_i$ to $i$-ty wiersz macierzy $M$).
"""
        )

        st.markdown("↓ Wybierz najlepsze k indeksów")
        st.markdown(
            r"""
Wybierz indeksy najlepszych $k$ wyników:

$$
\mathrm{indices} = \operatorname{argmax}_k(u)
$$
"""
        )

        st.markdown("↓ Użyj tych indeksów do lookup")
        top_k = 5
        # Example indices for demonstration (derived from u in a real system).
        indices = [2, 3, 8, 0, 9][:top_k]
        st.markdown(f"Przykład: **indeksy** = {indices}")

        lookup = toy.iloc[indices].copy()
        lookup.insert(0, "indeks", indices)
        lookup.insert(1, "zapytanie", query)
        st.dataframe(lookup[["indeks", "zapytanie", "passage_id", "text"]], width="stretch")

    st.subheader("Kalibracja: Hits@k vs k (wiki-trivia)")
    st.caption("Pole: hits_at_k. Źródło: .cache/calibration/wiki-trivia/test/hits_points_maxk200_log_p20.csv")

    cal = _load_calibration_hits_points()
    if cal.empty:
        st.info(
            "Nie znaleziono pliku kalibracji: .cache/calibration/wiki-trivia/test/hits_points_maxk200_log_p20.csv"
        )
    else:
        cal = cal[cal["method"].isin(["bm25", "tfidf"])].copy()
        cal["k"] = pd.to_numeric(cal["k"], errors="coerce")
        cal["hits_at_k"] = pd.to_numeric(cal["hits_at_k"], errors="coerce")
        cal = cal.dropna(subset=["k", "hits_at_k"]).sort_values(["method", "k"])

        # Plot metric (y) vs k (x). The calibration CSV only contains Hits@k curves,
        # so additional options are derived from the logged hits_at_k series.
        cal["hits_at_max_k"] = cal.groupby("method")["hits_at_k"].transform("max")
        cal["hits_norm_to_max_k"] = cal["hits_at_k"] / cal["hits_at_max_k"]
        cal["k_fraction_of_max"] = cal["k"] / cal["max_k"]
        cal["hits_per_k"] = cal["hits_at_k"] / cal["k"]

        cal = cal.sort_values(["method", "k"])
        cal["marginal_hits_gain"] = cal.groupby("method")["hits_at_k"].diff()
        cal["marginal_k"] = cal.groupby("method")["k"].diff()
        cal["marginal_hits_per_doc"] = cal["marginal_hits_gain"] / cal["marginal_k"]

        metric_label_to_col = {
            "hits_at_k": "hits_at_k",
            "hits_norm_to_max_k": "hits_norm_to_max_k",
            "hits_per_k": "hits_per_k",
            "marginal_hits_gain": "marginal_hits_gain",
            "marginal_hits_per_doc": "marginal_hits_per_doc",
            "k_fraction_of_max": "k_fraction_of_max",
        }
        selected_metric_label = st.selectbox(
            "Metryka (oś Y)",
            options=list(metric_label_to_col.keys()),
            index=0,
        )
        metric_col = metric_label_to_col[selected_metric_label]

        plot_df = cal[["method", "k", metric_col]].dropna(subset=[metric_col]).copy()

        legend_toggle = alt.selection_point(fields=["method"], bind="legend")
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("k:Q", title="k"),
                y=alt.Y(f"{metric_col}:Q", title=selected_metric_label),
                color=alt.Color(
                    "method:N",
                    title="metoda",
                    # User-requested palette: orange + white
                    scale=alt.Scale(domain=["bm25", "tfidf"], range=["#FF7F0E", "#FFFFFF"]),
                ),
                tooltip=[
                    alt.Tooltip("method:N"),
                    alt.Tooltip("k:Q"),
                    alt.Tooltip(f"{metric_col}:Q", format=".6f"),
                ],
            )
            .add_params(legend_toggle)
            .transform_filter(legend_toggle)
            .properties(height=320)
            .configure(background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="rgba(255,255,255,0.10)",
                tickColor="rgba(255,255,255,0.25)",
                domainColor="rgba(255,255,255,0.25)",
            )
            .configure_legend(
                labelColor="white",
                titleColor="white",
            )
        )
        st.altair_chart(chart, width="stretch")

        with st.expander("Co oznaczają te metryki kalibracji?"):
            if hasattr(st, "page_link"):
                st.page_link(
                    "pages/01_podsumowanie_problemu.py",
                    label="Otwórz: Podsumowanie problemu (wzory metryk)",
                )
            else:
                st.markdown("Pełne wzory metryk są na stronie **Podsumowanie problemu**.")

            st.markdown(
                """
Ten plik kalibracji zawiera **krzywą Hits@k** (dla każdej metody) i rysujemy wybraną metrykę względem $k$:

- **hits_at_k**: to samo Hits@k co w Podsumowaniu problemu, policzone dla różnych $k$.
- **hits_norm_to_max_k**: $\mathrm{Hits@k} / \max_{k'} \mathrm{Hits@k'}$ w obrębie metody (normalizacja do 1.0 dla najlepszego $k$).
- **hits_per_k**: $\mathrm{Hits@k} / k$ (prosty wskaźnik „trafień na pobrany dokument”).
- **marginal_hits_gain**: zmiana Hits@k między dwoma kolejnymi punktami $k$.
- **marginal_hits_per_doc**: $\Delta\mathrm{Hits@k} / \Delta k$ między punktami (nachylenie krzywej).
- **k_fraction_of_max**: $k / \mathrm{max\_k}$ (jak daleko jesteś w zakresie kalibracji).

Wszystkie metryki pochodne są liczone **wyłącznie z tego CSV**; jedyną „prawdziwą” metryką IR w tym pliku jest Hits@k.
"""
            )

        with st.expander("Pokaż tabelę kalibracji"):
            # Only show columns that are in the allow-list.
            cal_show = [c for c in ["method", "k", "hits_at_k"] if c in cal.columns]
            st.dataframe(cal[cal_show], width="stretch")

        st.markdown(
            """
Po co jest ta kalibracja:

- Pomaga dobrać, jak duże $k$ jest potrzebne, zanim zyski zaczną się wypłaszczać.
- To kompromis inżynierski: większe $k$ może poprawiać Hits@k, ale zwiększa czas i zużycie pamięci (szczególnie przy rerankingu).
"""
        )

    st.subheader("Najlepsza baza leksykalna (wg hits_at_k)")
    comparable = lexical.dropna(subset=["hits_at_k", "k"]).copy()
    if comparable.empty:
        st.info("Nie da się wybrać najlepszego uruchomienia: brakuje hits_at_k.")
        return

    # Ensure numeric compare
    comparable["hits_at_k"] = pd.to_numeric(comparable["hits_at_k"], errors="coerce")
    comparable["k"] = pd.to_numeric(comparable["k"], errors="coerce")

    # Logi zwykle mają k=10 dla obu metod; grupowanie po k zapobiega nieporównywalnym zestawieniom.
    best_rows: list[pd.Series] = []
    for k_val, group in comparable.groupby("k", dropna=False):
        group = group.dropna(subset=["hits_at_k"])
        if group.empty:
            continue
        best_rows.append(group.sort_values("hits_at_k", ascending=False).iloc[0])

    if not best_rows:
        st.info("Brak porównywalnych grup do wyboru najlepszego uruchomienia.")
        return

    best = pd.DataFrame(best_rows)
    st.dataframe(best[["method", "k", "hits_at_k"]], width="stretch")


page()
