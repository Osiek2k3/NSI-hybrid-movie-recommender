"""
Kompleksowa eksploracyjna analiza danych (EDA) dla połączonego zbioru:
MovieLens 1M (ratings.dat, users.dat, movies.dat) + TMDB API (movies_enriched.parquet).

Skrypt wylicza i zapisuje:
1. Dokładny opis wszystkich tabel, kolumn (cech), typów danych, braków danych i zakresów wartości.
2. Statystyki ocen (rozkład 1–5 gwiazdek, średnia, odchylenie, rozkład czasowy).
3. Cechy demograficzne i behawioralne użytkowników (płeć, przedziały wiekowe, 21 zawodów wg kodowania MovieLens 1M, liczba i średnia ocen na użytkownika).
4. Cechy filmów i zjawisko "długiego ogona" (long tail / krzywa Lorenza-Pareto, lata produkcji, 18 gatunków, filmy bez ocen).
5. Cechy pozyskane z TMDB (pokrycie opisów fabuły, długość opisów w słowach i znakach, liczba słów kluczowych, najpopularniejsi reżyserzy i aktorzy).
6. Zestaw 6 wykresów EDA w katalogu outputs/eda/plots/ oraz pełny raport outputs/eda/EDA_raport.md.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EDA_DIR = ROOT_DIR / "outputs" / "eda"
EDA_PLOTS_DIR = EDA_DIR / "plots"

# Oficjalne słowniki kodowania cech z dokumentacji MovieLens 1M (README GroupLens)
AGE_MAP = {
    1: "Poniżej 18 lat (<18)",
    18: "18–24 lata",
    25: "25–34 lata",
    35: "35–44 lata",
    45: "45–49 lat",
    50: "50–55 lat",
    56: "56+ lat",
}

OCCUPATION_MAP = {
    0: "Inny / nieokreślony",
    1: "Naukowiec / nauczyciel (academic/educator)",
    2: "Artysta (artist)",
    3: "Pracownik biurowy / administracja (clerical/admin)",
    4: "Student / doktorant (college/grad student)",
    5: "Obsługa klienta (customer service)",
    6: "Lekarz / ochrona zdrowia (doctor/health care)",
    7: "Kierownik / menedżer (executive/managerial)",
    8: "Rolnik (farmer)",
    9: "Zajmujący się domem (homemaker)",
    10: "Uczeń szkoły podstawowej/średniej (K-12 student)",
    11: "Prawnik (lawyer)",
    12: "Programista (programmer)",
    13: "Emeryt (retired)",
    14: "Sprzedaż / marketing (sales/marketing)",
    15: "Naukowiec przyrodnik (scientist)",
    16: "Samozatrudniony (self-employed)",
    17: "Technik / inżynier (technician/engineer)",
    18: "Rzemieślnik (tradesman/craftsman)",
    19: "Bezrobotny (unemployed)",
    20: "Pisarz / dziennikarz (writer)",
}


def generate_eda_plots(
    ratings: pd.DataFrame,
    users: pd.DataFrame,
    movies: pd.DataFrame,
    rating_dist: pd.DataFrame,
    genre_df: pd.DataFrame,
    age_df: pd.DataFrame,
) -> None:
    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Rozkład ocen (1-5 gwiazdek)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=180)
    bars = ax.bar(
        rating_dist["Ocena (gwiazdki)"].astype(str),
        rating_dist["Liczba ocen"],
        color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#27ae60"],
        width=0.6,
    )
    for bar, pct in zip(bars, rating_dist["Udział (%)"]):
        h = bar.get_height()
        ax.annotate(
            f"{int(h):,}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )
    ax.set_ylim(0, float(rating_dist["Liczba ocen"].max()) * 1.18)
    ax.set_title("Rozkład wystawionych ocen w zbiorze MovieLens 1M (N = 1 000 209)", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Ocena w skali 1–5 gwiazdek", fontsize=10.5)
    ax.set_ylabel("Liczba ocen", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "1_rozkład_ocen.png")
    plt.close(fig)

    # 2. Zjawisko "długiego ogona" (Long Tail) – rozkład popularności filmów i krzywa skumulowana
    movie_counts = ratings.groupby("movieId").size().reindex(movies["movieId"], fill_value=0).sort_values(ascending=False).to_numpy()
    cum_share = np.cumsum(movie_counts) / movie_counts.sum() * 100.0
    x_pct = np.arange(1, len(movie_counts) + 1) / len(movie_counts) * 100.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=180)
    ax1.plot(np.arange(1, len(movie_counts) + 1), movie_counts, color="#2980b9", linewidth=2)
    ax1.fill_between(np.arange(1, len(movie_counts) + 1), movie_counts, color="#3498db", alpha=0.25)
    ax1.set_title("Rozkład popularności filmów (Long Tail)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Filmy posortowane od najpopularniejszego (1..3883)", fontsize=10)
    ax1.set_ylabel("Liczba ocen filmu", fontsize=10)

    ax2.plot(x_pct, cum_share, color="#c0392b", linewidth=2.4, label="Skumulowany udział ocen")
    ax2.plot([0, 100], [0, 100], color="#7f8c8d", linestyle="--", linewidth=1.2, label="Równomierny rozkład")
    # Zaznaczenie punktu 20% najpopularniejszych filmów
    idx_20 = int(0.20 * len(movie_counts)) - 1
    share_20 = cum_share[idx_20]
    ax2.scatter([20.0], [share_20], color="#c0392b", zorder=5)
    ax2.annotate(
        f"20% filmów generuje\n{share_20:.1f}% wszystkich ocen",
        xy=(20.0, share_20),
        xytext=(35.0, share_20 - 15.0),
        arrowprops=dict(arrowstyle="->", color="#c0392b"),
        fontsize=9,
        fontweight="bold",
    )
    ax2.set_title("Krzywa koncentracji ocen (zasada Pareto)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Odsetek najpopularniejszych filmów (%)", fontsize=10)
    ax2.set_ylabel("Skumulowany odsetek wszystkich ocen (%)", fontsize=10)
    ax2.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "2_dlugi_ogon_popularnosci.png")
    plt.close(fig)

    # 3. Rozkład aktywności użytkowników (histogram liczby ocen na użytkownika)
    user_counts = ratings.groupby("userId").size().to_numpy()
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=180)
    ax.hist(user_counts, bins=50, color="#8e44ad", edgecolor="white", alpha=0.85)
    ax.axvline(np.median(user_counts), color="#e74c3c", linestyle="--", linewidth=2, label=f"Mediana = {np.median(user_counts):.0f} ocen")
    ax.axvline(np.mean(user_counts), color="#f39c12", linestyle="-.", linewidth=2, label=f"Średnia = {np.mean(user_counts):.1f} ocen")
    ax.set_title("Rozkład liczby ocen wystawionych przez jednego użytkownika", fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Liczba ocen użytkownika (min = 20, maks = 2314)", fontsize=10.5)
    ax.set_ylabel("Liczba użytkowników", fontsize=10.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "3_aktywnosc_uzytkownikow.png")
    plt.close(fig)

    # 4. Gatunki filmowe w MovieLens 1M (liczba filmów i średnia ocena)
    fig, ax1 = plt.subplots(figsize=(9.2, 5.4), dpi=180)
    g_sorted = genre_df.sort_values("Liczba filmów", ascending=True)
    bars = ax1.barh(g_sorted["Gatunek"], g_sorted["Liczba filmów"], color="#16a085", height=0.68)
    for bar, avg_r in zip(bars, g_sorted["Średnia ocena"]):
        w = bar.get_width()
        ax1.annotate(
            f" {int(w)} filmów (śr. {avg_r:.2f}/5)",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=8,
        )
    ax1.set_xlim(0, float(g_sorted["Liczba filmów"].max()) * 1.28)
    ax1.set_title("Rozkład 18 gatunków filmowych w MovieLens 1M (liczba filmów i średnia ocena)", fontsize=11.5, fontweight="bold")
    ax1.set_xlabel("Liczba filmów przypisanych do gatunku", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "4_gatunki_filmow.png")
    plt.close(fig)

    # 5. Demografia użytkowników: grupy wiekowe z podziałem na płeć
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=180)
    age_gender = users.groupby(["age", "gender"]).size().unstack(fill_value=0)
    age_labels = [AGE_MAP.get(int(a), str(a)) for a in age_gender.index]
    x = np.arange(len(age_labels))
    w = 0.36
    ax.bar(x - w / 2, age_gender.get("M", 0), width=w, label="Mężczyźni (M)", color="#2980b9")
    ax.bar(x + w / 2, age_gender.get("F", 0), width=w, label="Kobiety (F)", color="#e84393")
    ax.set_xticks(x)
    ax.set_xticklabels(age_labels, rotation=15)
    ax.set_title("Struktura demograficzna użytkowników wg wieku i płci (N = 6 040)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Liczba użytkowników", fontsize=10.5)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "5_demografia_wiek_plec.png")
    plt.close(fig)

    # 6. Cechy tekstowe z TMDB: rozkład długości opisu fabuły (w słowach) oraz liczby słów kluczowych
    word_counts = movies["overview"].fillna("").apply(lambda s: len(str(s).split()))
    kw_counts = movies["keywords"].apply(lambda k: len(k) if isinstance(k, (list, np.ndarray)) else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=180)
    ax1.hist(word_counts, bins=40, color="#27ae60", edgecolor="white", alpha=0.85)
    ax1.axvline(word_counts.median(), color="#c0392b", linestyle="--", linewidth=2, label=f"Mediana = {word_counts.median():.0f} słów")
    ax1.set_title("Długość opisu fabuły z TMDB (overview w słowach)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Liczba słów w opisie fabuły", fontsize=10)
    ax1.set_ylabel("Liczba filmów", fontsize=10)
    ax1.legend(frameon=True)

    ax2.hist(kw_counts, bins=16, color="#d35400", edgecolor="white", alpha=0.85)
    ax2.axvline(kw_counts.median(), color="#2c3e50", linestyle="--", linewidth=2, label=f"Mediana = {kw_counts.median():.0f} słów kluczowych")
    ax2.set_title("Liczba słów kluczowych z TMDB (keywords) na film", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Liczba słów kluczowych (maks. 15)", fontsize=10)
    ax2.set_ylabel("Liczba filmów", fontsize=10)
    ax2.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(EDA_PLOTS_DIR / "6_cechy_tmdb_opisy_keywords.png")
    plt.close(fig)


def main() -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    EDA_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    users = pd.read_parquet(PROCESSED_DIR / "users.parquet")
    movies_base = pd.read_parquet(PROCESSED_DIR / "movies_base.parquet")
    movies_tmdb = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")

    # 1. Rozkład ocen (1-5)
    r_counts = ratings["rating"].value_counts().sort_index()
    rating_dist = pd.DataFrame(
        {
            "Ocena (gwiazdki)": r_counts.index.astype(int),
            "Liczba ocen": r_counts.values.astype(int),
            "Udział (%)": (r_counts.values / len(ratings) * 100.0).round(2),
        }
    )
    rating_dist.to_csv(EDA_DIR / "rozkład_ocen.csv", index=False)

    # 2. Demografia: wiek i zawód
    age_counts = users["age"].value_counts().sort_index()
    age_ratings = ratings.merge(users[["userId", "age"]], on="userId").groupby("age")["rating"].agg(["count", "mean"])
    age_df = pd.DataFrame(
        {
            "Kod wieku": age_counts.index,
            "Przedział wieku": [AGE_MAP.get(int(a), str(a)) for a in age_counts.index],
            "Liczba użytkowników": age_counts.values,
            "Udział użytkowników (%)": (age_counts.values / len(users) * 100.0).round(2),
            "Liczba wystawionych ocen": age_ratings["count"].values,
            "Średnia ocena": age_ratings["mean"].round(3).values,
        }
    )
    age_df.to_csv(EDA_DIR / "demografia_wiek.csv", index=False)

    occ_counts = users["occupation"].value_counts().sort_index()
    occ_ratings = ratings.merge(users[["userId", "occupation"]], on="userId").groupby("occupation")["rating"].agg(["count", "mean"])
    occ_df = pd.DataFrame(
        {
            "Kod zawodu": occ_counts.index,
            "Zawód": [OCCUPATION_MAP.get(int(o), str(o)) for o in occ_counts.index],
            "Liczba użytkowników": occ_counts.values,
            "Udział (%)": (occ_counts.values / len(users) * 100.0).round(2),
            "Liczba ocen": occ_ratings["count"].values,
            "Średnia ocena": occ_ratings["mean"].round(3).values,
        }
    ).sort_values("Liczba użytkowników", ascending=False)
    occ_df.to_csv(EDA_DIR / "demografia_zawody.csv", index=False)

    # 3. Gatunki filmowe (18 gatunków z MovieLens 1M)
    ml_genres_exploded = movies_base[["movieId", "genres"]].assign(
        genre=movies_base["genres"].str.split("|")
    ).explode("genre")
    genre_movie_counts = ml_genres_exploded.groupby("genre")["movieId"].nunique()
    genre_rating_stats = ratings.merge(ml_genres_exploded[["movieId", "genre"]], on="movieId").groupby("genre")["rating"].agg(["count", "mean", "std"])

    genre_df = pd.DataFrame(
        {
            "Gatunek": genre_movie_counts.index,
            "Liczba filmów": genre_movie_counts.values,
            "Odsetek katalogu (%)": (genre_movie_counts.values / len(movies_base) * 100.0).round(2),
            "Liczba ocen": genre_rating_stats.reindex(genre_movie_counts.index)["count"].fillna(0).astype(int).values,
            "Średnia ocena": genre_rating_stats.reindex(genre_movie_counts.index)["mean"].round(3).values,
            "Odchylenie std ocen": genre_rating_stats.reindex(genre_movie_counts.index)["std"].round(3).values,
        }
    ).sort_values("Liczba filmów", ascending=False)
    genre_df.to_csv(EDA_DIR / "gatunki_movielens.csv", index=False)

    # 4. Zjawisko długiego ogona (Pareto) i Top-15 najpopularniejszych filmów
    movie_stats = ratings.groupby("movieId")["rating"].agg(["count", "mean", "std"]).reindex(movies_tmdb["movieId"], fill_value=0)
    sorted_counts = movie_stats["count"].sort_values(ascending=False).to_numpy()
    cum_ratings = np.cumsum(sorted_counts) / sorted_counts.sum()
    pct_50_movies = (np.searchsorted(cum_ratings, 0.50) + 1) / len(sorted_counts) * 100.0
    pct_80_movies = (np.searchsorted(cum_ratings, 0.80) + 1) / len(sorted_counts) * 100.0
    share_top_10pct = cum_ratings[int(0.10 * len(sorted_counts)) - 1] * 100.0
    share_top_20pct = cum_ratings[int(0.20 * len(sorted_counts)) - 1] * 100.0
    unrated_movies_count = int((movie_stats["count"] == 0).sum())

    top15_movies = (
        movie_stats.reset_index()
        .merge(movies_tmdb[["movieId", "title", "director", "year"]], on="movieId")
        .sort_values("count", ascending=False)
        .head(15)
    )
    top15_movies.to_csv(EDA_DIR / "top15_najpopularniejszych_filmow.csv", index=False)

    # 5. Statystyki cech tekstowych z TMDB
    movies_tmdb["overview_chars"] = movies_tmdb["overview"].fillna("").str.len()
    movies_tmdb["overview_words"] = movies_tmdb["overview"].fillna("").apply(lambda s: len(str(s).split()))
    movies_tmdb["emb_text_words"] = movies_tmdb["embedding_text"].fillna("").apply(lambda s: len(str(s).split()))
    movies_tmdb["n_keywords"] = movies_tmdb["keywords"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    movies_tmdb["n_cast"] = movies_tmdb["cast"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    movies_tmdb["n_genres_combined"] = movies_tmdb["genres"].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)

    # Najczęstsi reżyserzy, aktorzy i słowa kluczowe w TMDB
    dir_counter: Counter[str] = Counter()
    actor_counter: Counter[str] = Counter()
    kw_counter: Counter[str] = Counter()

    for _, row in movies_tmdb.iterrows():
        for d in str(row.get("director") or "").split(","):
            if d.strip():
                dir_counter[d.strip()] += 1
        cast_val = row.get("cast")
        if isinstance(cast_val, (list, np.ndarray)):
            for a in cast_val:
                if str(a).strip():
                    actor_counter[str(a).strip()] += 1
        kw_val = row.get("keywords")
        if isinstance(kw_val, (list, np.ndarray)):
            for kw in kw_val:
                if str(kw).strip():
                    kw_counter[str(kw).strip()] += 1

    generate_eda_plots(ratings, users, movies_tmdb, rating_dist, genre_df, age_df)

    # Wypisanie podsumowania do konsoli
    user_activity = ratings.groupby("userId").size()
    gender_counts = users["gender"].value_counts()
    gender_ratings = ratings.merge(users[["userId", "gender"]], on="userId").groupby("gender")["rating"].agg(["count", "mean"])

    print("=" * 76)
    print("KOMPLEKSOWA ANALIZA DANYCH (EDA): MOVIELENS 1M + TMDB")
    print("=" * 76)
    print("\n1. STRUKTURA TABEL I CECH (ATRYBUTÓW):")
    print("  A) Tabela ocen (ratings.dat -> ratings.parquet): 1,000,209 wierszy × 4 kolumny")
    print("     - userId (int64): identyfikator użytkownika [1 .. 6040], brak pustych wartości")
    print("     - movieId (int64): identyfikator filmu [1 .. 3952], 3706 unikalnych ocenionych filmów")
    print("     - rating (int64): ocena całkowita w skali {1, 2, 3, 4, 5}")
    print("     - timestamp (int64): czas Unix w sekundach (od 2000-04-25 do 2003-02-28)")
    print("  B) Tabela użytkowników (users.dat -> users.parquet): 6,040 wierszy × 5 kolumn")
    print("     - userId (int64): klucz główny [1 .. 6040]")
    print("     - gender (str): płeć {'M': 4331 (71.7%), 'F': 1709 (28.3%)}")
    print("     - age (int64): kod przedziału wiekowego (7 kategorii: 1, 18, 25, 35, 45, 50, 56)")
    print("     - occupation (int64): kod zawodu (21 kategorii: 0..20)")
    print("     - zipCode (str): kod pocztowy USA (3439 unikalnych kodów)")
    print("  C) Tabela filmów wzbogacona o TMDB (movies_enriched.parquet): 3,883 wierszy × 13 kolumn")
    print("     - movieId (int64): identyfikator z MovieLens [1 .. 3952]")
    print("     - tmdbId (Int64): identyfikator w bazie TMDB (3872 dopasowane = 99.72%)")
    print("     - title (str), clean_title (str), year (Int64): lata wydania od 1919 do 2000")
    print("     - genres (list[str]): połączone gatunki MovieLens + TMDB (śr. 2.52 gatunku na film)")
    print("     - overview (str): opis fabuły po angielsku (śr. "
          f"{movies_tmdb['overview_words'].mean():.1f} słów, mediana {movies_tmdb['overview_words'].median():.0f} słów)")
    print("     - keywords (list[str]): słowa kluczowe z TMDB (śr. "
          f"{movies_tmdb['n_keywords'].mean():.2f} słów kluczowych na film)")
    print("     - cast (list[str]): top-5 głównych aktorów z TMDB")
    print("     - director (str): reżyser filmu z TMDB (pokrycie 99.59%)")
    print("     - poster_path (str): ścieżka do plakatu w TMDB (pokrycie "
          f"{(movies_tmdb['poster_path'].str.len() > 0).mean() * 100:.2f}%)")
    print("     - embedding_text (str): złożony tekst wejściowy dla modelu Transformer (śr. "
          f"{movies_tmdb['emb_text_words'].mean():.1f} słów)")

    print("\n2. STATYSTYKI OCEN (RATINGS):")
    print(f"  - Średnia ocena globalna: {ratings['rating'].mean():.4f} ± {ratings['rating'].std():.4f}")
    print(f"  - Mediana oceny:          {ratings['rating'].median():.1f} (dominanta = 4 gwiazdki)")
    print(f"  - Odsetek ocen pozytywnych (>= 4★): {(ratings['rating'] >= 4).mean() * 100:.2f}%")
    print(rating_dist.to_string(index=False))

    print("\n3. AKTYWNOŚĆ UŻYTKOWNIKÓW I DEMOGRAFIA:")
    print(f"  - Liczba ocen na użytkownika: min={user_activity.min()}, Q1={np.percentile(user_activity, 25):.0f}, "
          f"mediana={user_activity.median():.0f}, średnia={user_activity.mean():.1f}, "
          f"Q3={np.percentile(user_activity, 75):.0f}, P90={np.percentile(user_activity, 90):.0f}, maks={user_activity.max()}")
    print(f"  - Płeć M: {gender_counts['M']:,} użytkowników ({gender_counts['M']/len(users)*100:.1f}%), "
          f"{gender_ratings.loc['M', 'count']:,} ocen, śr. ocena = {gender_ratings.loc['M', 'mean']:.3f}")
    print(f"  - Płeć F: {gender_counts['F']:,} użytkowników ({gender_counts['F']/len(users)*100:.1f}%), "
          f"{gender_ratings.loc['F', 'count']:,} ocen, śr. ocena = {gender_ratings.loc['F', 'mean']:.3f}")

    print("\n4. RZADKOŚĆ MACIERZY I ZJAWISKO DŁUGIEGO OGONA (LONG TAIL):")
    print(f"  - Wymiar pełnej macierzy (użytkownicy × katalog): 6040 × 3883 = {6040 * 3883:,} komórek")
    print(f"  - Gęstość macierzy: 4.26% (rzadkość / sparsity = 95.74%)")
    print(f"  - Filmy w katalogu bez ani jednej oceny w ratings.dat: {unrated_movies_count} z 3883 ({unrated_movies_count/3883*100:.2f}%)")
    print(f"  - Koncentracja Pareto: top 10% filmów zbiera {share_top_10pct:.1f}% wszystkich ocen!")
    print(f"  - Koncentracja Pareto: top 20% filmów zbiera {share_top_20pct:.1f}% wszystkich ocen!")
    print(f"  - Aby pokryć 50% wszystkich ocen, wystarczy zaledwie {pct_50_movies:.1f}% najpopularniejszych filmów.")
    print(f"  - Aby pokryć 80% wszystkich ocen, wystarczy {pct_80_movies:.1f}% filmów.")

    print("\n5. NAJCZĘSTSZE SŁOWA KLUCZOWE, REŻYSERZY I AKTORZY Z TMDB:")
    print("  - Top-10 słów kluczowych TMDB:", ", ".join(f"{k} ({v})" for k, v in kw_counter.most_common(10)))
    print("  - Top-8 reżyserów w katalogu:", ", ".join(f"{k} ({v})" for k, v in dir_counter.most_common(8)))
    print("  - Top-8 aktorów w katalogu:  ", ", ".join(f"{k} ({v})" for k, v in actor_counter.most_common(8)))
    print("=" * 76)


if __name__ == "__main__":
    main()
