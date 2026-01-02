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
    st.title("Podsumowanie końcowe")
    st.caption("Najlepsza metoda jest wybierana na podstawie hits_at_k, wśród porównywalnych uruchomień.")

    st.markdown(
        """
### Jak czytać to podsumowanie

- Na tej stronie traktujemy **Hits@k** jako główny sygnał typu „czy znaleźliśmy cokolwiek trafnego?”.
- Metryki wrażliwe na pozycję (MRR/nDCG) są nadal przydatne, bo pomagają rozróżnić:
    „więcej trafień gdziekolwiek w top‑k” vs „lepsze ułożenie na samym szczycie listy”.
"""
    )

    st.subheader("Interpretacja: co zwykle się dzieje")
    st.markdown(
        """
- **Retrieval leksykalny jest szybki**: TF‑IDF/BM25 są rzadkie (sparse), dobrze działają z cache i dobrze skalują się
  jako pierwszy etap.
- **Reranking ma sens jako lejek**: metody leksykalne tanio zawężają przestrzeń wyszukiwania do małego zbioru kandydatów,
  a dokładniejszy model przestawia kolejność tylko wśród tych kandydatów.
- **Reranker może być inny**: w zarejestrowanych uruchomieniach użyto bi-encodera, ale ten sam schemat „lejka” działa też
  z innymi rerankerami (np. cross-encoder, LSA/SVD, learned rankers), o ile punktują ograniczoną listę kandydatów.

Jeśli metoda hybrydowa zwiększa `hits_at_k`, zwykle oznacza to, że lejek częściej „dowozi” przynajmniej jeden trafny
passage — nawet jeśli metryki wrażliwe na pozycję nie zawsze rosną w tym samym stopniu.
"""
    )

    df = _wiki_trivia_only(_load_all_metrics())
    if df.empty:
        st.error("Nie znaleziono metryk dla wiki-trivia w .cache/submissions/**/metrics.csv")
        return

    methods = [
        "tfidf_cosine",
        "bm25_okapi",
        "hybrid_bm25_biencoder",
        "hybrid_tfidf_biencoder",
    ]
    df = df[df["method"].isin(methods)].copy()
    if df.empty:
        st.error("Brak uruchomień kwalifikujących się do podsumowania.")
        return

    df["hits_at_k"] = pd.to_numeric(df["hits_at_k"], errors="coerce")
    df["ndcg_at_k"] = pd.to_numeric(df["ndcg_at_k"], errors="coerce")
    df["k"] = pd.to_numeric(df["k"], errors="coerce")

    st.subheader("Porównywalne uruchomienia (wiki-trivia)")
    show_cols = [
        "method",
        "dataset_id",
        "subdataset",
        "questions_split",
        "pairs_split",
        "k",
        "n_questions",
        "n_labeled",
        "hits_at_k",
        "ndcg_at_k",
        "recall_at_k",
        "precision_at_k",
        "mrr_at_k",
        "top_k_candidates",
        "rerank_k",
        "biencoder_batch_size",
        "biencoder_max_length",
        "bm25_k1",
        "bm25_b",
        "chunk_size",
    ]
    st.dataframe(
        df[show_cols].sort_values(["k", "hits_at_k"], ascending=[True, False]),
        width="stretch",
    )

    st.subheader("Porównaj metrykę")
    metric_options = [
        "hits_at_k",
        "ndcg_at_k",
        "mrr_at_k",
        "recall_at_k",
        "precision_at_k",
    ]
    selected_metric = st.selectbox(
        "Metryka",
        options=metric_options,
        index=0,
    )
    k_values = sorted([k for k in df["k"].dropna().unique().tolist()])
    selected_k = st.selectbox("k", options=k_values) if k_values else None

    if selected_k is not None:
        plot_df = df[df["k"] == selected_k].copy()
        plot_df[selected_metric] = pd.to_numeric(plot_df[selected_metric], errors="coerce")
        plot_df = plot_df.dropna(subset=[selected_metric])

        chart = (
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "method:N",
                    title="metoda",
                    sort=methods,
                    axis=alt.Axis(labelAngle=0),
                ),
                y=alt.Y(f"{selected_metric}:Q", title=selected_metric),
                color=alt.value("#FF7F0E"),
                tooltip=[
                    alt.Tooltip("method:N"),
                    alt.Tooltip("k:Q"),
                    alt.Tooltip(f"{selected_metric}:Q", format=".6f"),
                ],
            )
            .properties(height=320)
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
    else:
        st.info("Wybierz k, aby narysować wykres.")

    with st.expander("Co oznaczają te metryki?"):
        # Streamlit multipage link (relative to app/main.py)
        if hasattr(st, "page_link"):
            st.page_link(
                "pages/01_podsumowanie_problemu.py",
                label="Otwórz: Podsumowanie problemu (wzory metryk)",
            )
        else:
            st.markdown("Pełne wzory metryk są na stronie **Podsumowanie problemu**.")

        st.markdown(
            """
    - **Hits@k**: czy co najmniej jeden trafny passage pojawia się gdziekolwiek w top‑$k$.
    - **Recall@k**: odsetek trafnych passage’ów odzyskanych w top‑$k$ (dla pojedynczego pytania).
    - **Precision@k**: odsetek trafnych passage’ów wśród top‑$k$ zwróconych.
    - **MRR@k**: premiuje umieszczenie pierwszego trafnego passage’a jak najwyżej (1/rank), ucięte do $k$.
    - **nDCG@k**: metryka wrażliwa na pozycję; premiuje trafne passage’y blisko góry listy; normalizowana per pytanie.

    Wszystkie są liczone dla wybranego $k$, a potem uśredniane po pytaniach z etykietami.
"""
        )

    st.subheader("Najlepsza metoda (wg hits_at_k)")
    comparable = df.dropna(subset=["hits_at_k", "k"]).copy()
    if comparable.empty:
        st.info("Nie da się wybrać najlepszej: brakuje hits_at_k albo k.")
        return

    comparable["hits_at_k"] = pd.to_numeric(comparable["hits_at_k"], errors="coerce")

    best_rows = []
    for k_val, group in comparable.groupby("k", dropna=False):
        group = group.dropna(subset=["hits_at_k"])
        if group.empty:
            continue
        best_rows.append(group.sort_values("hits_at_k", ascending=False).iloc[0])

    best = pd.DataFrame(best_rows)
    st.dataframe(best[["k", "method", "hits_at_k"]], width="stretch")

    st.subheader("Kompromisy widoczne w logach")
    st.markdown(
        """
- **Bazowe metody leksykalne** mają mniej „gałek” w logach (np. BM25 ma `bm25_k1`, `bm25_b`).
- **Uruchomienia hybrydowe** wprowadzają pipeline 2‑etapowy z dodatkowymi parametrami:
    - Rozmiar etapu 1: `top_k_candidates`
    - Rozmiar rerankingu: `rerank_k`
    - Wykonanie bi-encodera: `biencoder_device`, `biencoder_batch_size`, `biencoder_max_length`

Ta strona pokazuje tylko to, co jest bezpośrednio zapisane w `metrics.csv`.
"""
    )

    st.subheader("Uwaga: cache robi tu robotę")
    st.markdown(
        """
Ta prezentacja opiera się na cache, dzięki czemu eksperymenty da się uruchamiać w praktyce:

- Artefakty leksykalne dla passage’ów są cache’owane w `.cache/preprocessed_data/...`.
- Wyniki ewaluacji są cache’owane w `.cache/submissions/.../metrics.csv` — i to właśnie czyta ta aplikacja.

Bez cache zarówno „wektoryzacja milionów passage’ów”, jak i „wiele wariantów eksperymentów” byłyby zbyt wolne.
"""
    )


page()
