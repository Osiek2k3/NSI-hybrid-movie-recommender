"""
Interaktywny prototyp systemu rekomendacji filmów (MovieLens 1M + TMDB) w Streamlit.
Zawiera:
1. Symulator Cold Start i Onboardingu (wybór 3–5 ulubionych filmów z 30 polaryzujących hitów
   lub dowolnych filmów z katalogu, wizualizacja wagi α(N), plakaty TMDB i wyjaśnienia po polsku).
2. Profil istniejącego użytkownika z MovieLens 1M (porównanie Hybrydy, ALS i Content-Based + ocena SVD).
3. Dashboard z wykresami i tabelami wyników eksperymentalnych.
4. Wymaganą przez licencję atrybucję TMDB (logo + informacja licencyjna).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation import create_main_split_with_activity_groups
from src.explanations import RecommendationExplainer
from src.models import (
    ALSRecommender,
    BayesianPopularRecommender,
    ContentBasedRecommender,
    DiversePopularRecommender,
    HybridRecommender,
    MostPopularRecommender,
    SVDRecommender,
)
from src.run_experiments import select_onboarding_candidates

ROOT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"
TMDB_LOGO_URL = (
    "https://www.themoviedb.org/assets/2/v4/logos/v2/"
    "blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg"
)


st.set_page_config(
    page_title="Hybrydowy System Rekomendacji Filmów (MovieLens 1M + TMDB)",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource(show_spinner="Wczytywanie danych i modeli rekomendacyjnych...")
def load_system():
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    movies = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    embeddings = np.load(PROCESSED_DIR / "movie_embeddings.npy")
    movie_ids = np.load(PROCESSED_DIR / "movie_ids.npy")

    split = create_main_split_with_activity_groups(ratings, test_ratio=0.2, seed=42)

    most_pop = MostPopularRecommender().fit(split.train_df)
    bayes_pop = BayesianPopularRecommender(m_prior=200.0).fit(split.train_df)
    diverse_pop = DiversePopularRecommender().fit(split.train_df, movies)
    cb_model = ContentBasedRecommender(embeddings, movie_ids).fit(split.train_df)
    als_model = ALSRecommender(
        movie_ids, factors=64, regularization=0.05, alpha_conf=15.0, iterations=20, random_state=42
    ).fit(split.train_df)
    svd_model = SVDRecommender(movie_ids, factors=64, epochs=12, random_state=42).fit(split.train_df)

    hybrid_model = HybridRecommender(
        cb_model=cb_model,
        als_model=als_model,
        user_train_counts=split.user_train_counts,
        k_param=20.0,
    )
    explainer = RecommendationExplainer(movies, embeddings, movie_ids)
    onboarding_30 = select_onboarding_candidates(ratings, movies, n_candidates=30)

    return {
        "ratings": ratings,
        "movies": movies.set_index("movieId", drop=False),
        "embeddings": embeddings,
        "movie_ids": movie_ids,
        "split": split,
        "most_pop": most_pop,
        "bayes_pop": bayes_pop,
        "diverse_pop": diverse_pop,
        "cb_model": cb_model,
        "als_model": als_model,
        "svd_model": svd_model,
        "hybrid_model": hybrid_model,
        "explainer": explainer,
        "onboarding_30": onboarding_30,
    }


def render_movie_card(movie_row: pd.Series, explanation_text: str | None = None, svd_pred: float | None = None):
    col_img, col_info = st.columns([1, 3])
    with col_img:
        poster = str(movie_row.get("poster_path") or "").strip()
        if poster:
            st.image(f"{TMDB_IMAGE_BASE}{poster}", use_container_width=True)
        else:
            st.markdown("🎞️ *Brak plakatu*")
    with col_info:
        st.markdown(f"#### {movie_row['title']}")
        genres = movie_row.get("genres")
        if isinstance(genres, (list, np.ndarray)):
            genres_str = ", ".join(str(g) for g in genres)
        else:
            genres_str = str(genres)
        director = str(movie_row.get("director") or "Nieznany")
        st.caption(f"**Gatunki:** {genres_str} | **Reżyser:** {director}")
        if svd_pred is not None:
            st.markdown(f"⭐ **Przewidywana ocena (SVD):** `{svd_pred:.2f} / 5.00`")
        if explanation_text:
            st.info(f"💡 **Wyjaśnienie AI:** {explanation_text}")
        overview = str(movie_row.get("overview") or "")
        if overview:
            with st.expander("Opis fabuły (TMDB)"):
                st.write(overview)


def main():
    st.title("🎬 Hybrydowy System Rekomendacji Filmów")
    st.markdown(
        "**MovieLens 1M + TMDB** | Łączenie embeddingów **Transformer (`all-MiniLM-L6-v2`)** "
        "z filtrowaniem kolaboratywnym **ALS** oraz **SVD**"
    )

    data = load_system()
    movies_df: pd.DataFrame = data["movies"]
    explainer: RecommendationExplainer = data["explainer"]

    tab1, tab2, tab3 = st.tabs(
        [
            "🚀 Cold Start i Onboarding (Nowy użytkownik)",
            "👤 Istniejący użytkownicy (MovieLens 1M)",
            "📊 Wyniki eksperymentów i wykresy",
        ]
    )

    with tab1:
        st.subheader("Symulacja nowego użytkownika (Cold Start: Etap 1 i Etap 2)")
        st.write(
            "Wybierz od **0 do 10 ulubionych filmów** z poniższej listy 30 znanych, zróżnicowanych gatunkowo "
            "i polaryzujących widzów tytułów. Zobaczysz, jak zmienia się waga **α(N)** oraz lista polecanych filmów."
        )

        onboarding_30: pd.DataFrame = data["onboarding_30"]
        options_map = {
            f"{row['title']} [{'/'.join(row['genres'][:2]) if isinstance(row['genres'], (list, np.ndarray)) else row['genres']}]": int(row["movieId"])
            for _, row in onboarding_30.iterrows()
        }

        selected_labels = st.multiselect(
            "Wskaż filmy, które lubisz (Onboarding):",
            options=list(options_map.keys()),
            default=list(options_map.keys())[:3],
        )
        liked_ids = [options_map[lbl] for lbl in selected_labels]
        n_selected = len(liked_ids)

        k_param = st.slider("Parametr k we wzorze α(N) = N / (N + k):", min_value=5, max_value=50, value=20, step=5)
        alpha_val = n_selected / (n_selected + k_param) if n_selected > 0 else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Liczba wybranych filmów (N)", f"{n_selected}")
        c2.metric("Udział treści (1 - α)", f"{(1.0 - alpha_val) * 100:.1f}%")
        c3.metric("Udział CF / ALS (α)", f"{alpha_val * 100:.1f}%")

        if n_selected == 0:
            st.warning(
                "**Etap 1 Cold Start (N = 0, zero informacji):** Użytkownik nie wybrał jeszcze żadnego filmu. "
                "Porównujemy strategie niepersonalizowane."
            )
            strategy = st.radio(
                "Wybierz strategię dla N = 0:",
                [
                    "Popularne i dobrze oceniane (Średnia bayesowska)",
                    "Popularne i zróżnicowane gatunkowo",
                    "MostPopular (najczęściej oceniane)",
                ],
                horizontal=True,
            )
            if "bayesowska" in strategy:
                recs = data["bayes_pop"].recommend(0, set(), k=6)
            elif "zróżnicowane" in strategy:
                recs = data["diverse_pop"].recommend(0, set(), k=6)
            else:
                recs = data["most_pop"].recommend(0, set(), k=6)

            for mid in recs:
                render_movie_card(movies_df.loc[mid])
                st.divider()
        else:
            hybrid_cs = HybridRecommender(
                cb_model=data["cb_model"],
                als_model=data["als_model"],
                user_train_counts={},
                k_param=float(k_param),
            )
            seen_set = set(liked_ids)
            recs_hybrid = hybrid_cs.recommend_cold_start(liked_ids, seen_set, k=6)

            st.markdown("### Polecane filmy (Model Hybrydowy)")
            for mid in recs_hybrid:
                exp = explainer.explain(
                    user_id=0,
                    liked_movie_ids=liked_ids,
                    recommended_movie_id=mid,
                    alpha_cf=alpha_val,
                    predicted_svd_rating=data["svd_model"].predict_rating(0, mid),
                )
                render_movie_card(
                    movies_df.loc[mid],
                    explanation_text=exp.explanation_pl,
                    svd_pred=exp.predicted_svd_rating,
                )
                st.divider()

    with tab2:
        st.subheader("Analiza rekomendacji dla użytkowników ze zbioru MovieLens 1M")
        split = data["split"]
        bucket_choice = st.selectbox(
            "Wybierz grupę aktywności użytkownika:",
            ["1-5", "6-20", "21-100", ">100"],
            index=1,
        )
        matching_users = sorted(
            [u for u, b in split.user_buckets.items() if b == bucket_choice]
        )
        selected_uid = st.selectbox("Wybierz ID użytkownika:", matching_users[:100])

        user_hist = split.train_df[split.train_df["userId"] == selected_uid].sort_values(
            "rating", ascending=False
        )
        n_hist = len(user_hist)
        alpha_u = data["hybrid_model"].compute_alpha(n_hist)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Liczba ocen w historii treningowej", f"{n_hist}")
        col_b.metric("Waga α (udział ALS)", f"{alpha_u * 100:.1f}%")
        col_c.metric("Udział treści (1 - α)", f"{(1.0 - alpha_u) * 100:.1f}%")

        with st.expander("Pokaż historię ocen użytkownika w zbiorze treningowym"):
            hist_display = user_hist.merge(
                movies_df[["movieId", "title", "director"]], on="movieId", how="left"
            )
            st.dataframe(hist_display[["movieId", "title", "rating", "director"]], use_container_width=True)

        seen_u = split.train_seen.get(selected_uid, set())
        liked_u = user_hist[user_hist["rating"] >= 4]["movieId"].astype(int).tolist()
        if not liked_u:
            liked_u = user_hist["movieId"].astype(int).tolist()

        recs_u = data["hybrid_model"].recommend(selected_uid, seen_u, k=5)
        st.markdown("### Top-5 rekomendacji hybrydowych wraz z wyjaśnieniem")
        for mid in recs_u:
            svd_r = data["svd_model"].predict_rating(selected_uid, mid)
            exp = explainer.explain(
                user_id=selected_uid,
                liked_movie_ids=liked_u,
                recommended_movie_id=mid,
                alpha_cf=alpha_u,
                predicted_svd_rating=svd_r,
            )
            render_movie_card(movies_df.loc[mid], explanation_text=exp.explanation_pl, svd_pred=svd_r)
            with st.expander("Pokaż wygenerowany prompt z faktami dla LLM"):
                st.code(exp.prompt_for_llm, language="markdown")
            st.divider()

    with tab3:
        st.subheader("Wyniki eksperymentów badawczych")
        if (PLOTS_DIR / "cold_start_ndcg_vs_N.png").exists():
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.image(str(PLOTS_DIR / "cold_start_ndcg_vs_N.png"), use_container_width=True)
            with col_p2:
                st.image(str(PLOTS_DIR / "activity_groups_ndcg.png"), use_container_width=True)
            st.image(str(PLOTS_DIR / "overall_models_ndcg.png"), use_container_width=True)
        else:
            st.info("Uruchom `python -m src.run_experiments`, aby wygenerować wykresy i tabele.")

    st.markdown("---")
    col_logo, col_attr = st.columns([1, 8])
    with col_logo:
        st.image(TMDB_LOGO_URL, width=90)
    with col_attr:
        st.caption(
            "**Atrybucja TMDB:** This product uses the TMDB API but is not endorsed or certified by TMDB. "
            "Dane ocen pochodzą ze zbioru MovieLens 1M (F. M. Harper, J. A. Konstan, *The MovieLens Datasets: "
            "History and Context*, ACM TiiS, 2015, GroupLens Research)."
        )


if __name__ == "__main__":
    main()
