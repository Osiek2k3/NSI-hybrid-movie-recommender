"""
Główny skrypt eksperymentalny realizujący wszystkie badania opisane w założeniach projektu:
1. Porównanie ogólne metod (MostPopular, Content-Based, SVD, ALS, Hybryda stała α=0.5, Hybryda dynamiczna α(n))
   oraz porównanie SVD vs ALS (RMSE/MAE kontra NDCG@10, Precision@10, Recall@10).
2. Badanie wpływu długości historii użytkownika w 4 grupach aktywności:
   1-5 ocen, 6-20 ocen, 21-100 ocen, >100 ocen.
3. Eksperyment Cold Start:
   - Etap 1 (N = 0, zero informacji): MostPopular vs Średnia bayesowska vs Popularne i zróżnicowane gatunkowo
   - Etap 2 (Onboarding dla N in {0, 1, 3, 5, 10}, uśrednione po 3 losowaniach nowych użytkowników):
     A. MostPopular, B. CF (ALS fold-in), C. Embeddingi Transformer, D. Hybryda dynamiczna α(N)
   - Wybór 30 polaryzujących i znanych filmów z różnych gatunków do formularza onboardingu.
4. Generowanie wykresów i tabel wynikowych do folderu outputs/.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import (
    create_main_split_with_activity_groups,
    evaluate_recommendations,
    mae,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)
from src.models import (
    ALSRecommender,
    BayesianPopularRecommender,
    ContentBasedRecommender,
    DiversePopularRecommender,
    HybridRecommender,
    MostPopularRecommender,
    SVDRecommender,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"


def evaluate_by_activity_groups(
    models_dict: dict[str, object],
    test_relevant: dict[int, set[int]],
    train_seen: dict[int, set[int]],
    user_buckets: dict[int, str],
    k: int = 10,
) -> pd.DataFrame:
    """Oblicza metryki NDCG@10, Precision@10, Recall@10 osobno dla każdej grupy aktywności."""
    buckets_order = ["1-5", "6-20", "21-100", ">100"]
    rows: list[dict[str, object]] = []

    for bucket in buckets_order:
        bucket_users = {
            u: rel
            for u, rel in test_relevant.items()
            if user_buckets.get(int(u)) == bucket and len(rel) > 0
        }
        for model_name, model_obj in models_dict.items():
            metrics = evaluate_recommendations(
                model_obj.recommend, bucket_users, train_seen, k=k
            )
            rows.append(
                {
                    "Grupa aktywności": bucket,
                    "Liczba użytkowników": metrics["n_eval_users"],
                    "Model": model_name,
                    "NDCG@10": metrics["NDCG@10"],
                    "Precision@10": metrics["Precision@10"],
                    "Recall@10": metrics["Recall@10"],
                }
            )

    return pd.DataFrame(rows)


def run_cold_start_experiment(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    embeddings: np.ndarray,
    movie_ids: np.ndarray,
    n_values: list[int] = [0, 1, 3, 5, 10],
    seeds: list[int] = [42, 101, 202],
    cold_user_ratio: float = 0.15,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Symulacja nowych użytkowników (Cold Start) dokładnie wg procedury z założeń projektu:
    1. Losujemy 15% użytkowników jako 'nowych' i całkowicie usuwamy ich z danych treningowych CF.
    2. Jako 'wybór przy rejestracji' (onboarding) bierzemy ich N najwcześniejszych filmów ocenionych na 4 lub 5.
    3. Pozostałe oceny tych użytkowników służą do sprawdzenia trafności rekomendacji.
    4. Powtarzamy dla N = 0, 1, 3, 5 i 10 oraz dla kilku losowań, a wyniki uśredniamy.
    """
    sorted_ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    all_users = sorted_ratings["userId"].unique()

    stage1_records: list[dict[str, object]] = []
    onboarding_records: list[dict[str, object]] = []

    for run_idx, seed in enumerate(seeds, start=1):
        print(f"  -> Losowanie Cold Start {run_idx}/{len(seeds)} (seed={seed}) ...")
        rng = np.random.default_rng(seed)
        n_cold = int(round(len(all_users) * cold_user_ratio))
        cold_users = set(rng.choice(all_users, size=n_cold, replace=False))

        warm_train_df = sorted_ratings[~sorted_ratings["userId"].isin(cold_users)].reset_index(drop=True)
        cold_df = sorted_ratings[sorted_ratings["userId"].isin(cold_users)].reset_index(drop=True)

        # Uczymy modele na użytkownikach 'warm' (bez udziału nowych użytkowników!)
        most_pop = MostPopularRecommender().fit(warm_train_df)
        bayes_pop = BayesianPopularRecommender(m_prior=200.0).fit(warm_train_df)
        diverse_pop = DiversePopularRecommender().fit(warm_train_df, movies)

        cb_model = ContentBasedRecommender(embeddings, movie_ids).fit(warm_train_df)
        als_model = ALSRecommender(
            movie_ids=movie_ids,
            factors=64,
            regularization=0.05,
            alpha_conf=15.0,
            iterations=20,
            random_state=seed,
        ).fit(warm_train_df)

        # Przygotowujemy chronologiczne pozytywne oceny każdego nowego użytkownika
        cold_user_pos: dict[int, list[int]] = {}
        for uid, grp in cold_df.groupby("userId", sort=False):
            pos_movies = grp[grp["rating"] >= 4]["movieId"].astype(int).tolist()
            # Wymagamy co najmniej 15 pozytywnych ocen, aby przy N=10 zostało min. 5 ocen do testu
            if len(pos_movies) >= 15:
                cold_user_pos[int(uid)] = pos_movies

        # Etap 1: N = 0 (porównanie 3 strategii "w ciemno")
        for s1_name, s1_model in [
            ("MostPopular (liczba ocen)", most_pop),
            ("Średnia bayesowska (popularne i wysoko oceniane)", bayes_pop),
            ("Popularne i zróżnicowane gatunkowo", diverse_pop),
        ]:
            ndcgs, precs, recalls = [], [], []
            for uid, pos_list in cold_user_pos.items():
                # Aby zbiór testowy był identyczny dla każdego N, oceniamy na filmach po pierwszych 10 pozytywnych
                rel_set = set(pos_list[10:])
                recs = s1_model.recommend(uid, set(), k=k)
                ndcgs.append(ndcg_at_k(recs, rel_set, k))
                precs.append(precision_at_k(recs, rel_set, k))
                recalls.append(recall_at_k(recs, rel_set, k))
            stage1_records.append(
                {
                    "seed": seed,
                    "Wariant": s1_name,
                    "NDCG@10": float(np.mean(ndcgs)),
                    "Precision@10": float(np.mean(precs)),
                    "Recall@10": float(np.mean(recalls)),
                }
            )

        # Etap 2: Onboarding dla N in {0, 1, 3, 5, 10}
        # W hybrydzie dla bardzo krótkiego onboardingu (N=1..10) używamy k_param=15 (przewaga treści na starcie)
        hybrid_cs = HybridRecommender(
            cb_model=cb_model,
            als_model=als_model,
            user_train_counts={},
            k_param=15.0,
        )

        for n_val in n_values:
            metrics_acc: dict[str, list[tuple[float, float, float]]] = {
                "A. MostPopular": [],
                "B. CF (ALS fold-in)": [],
                "C. Embeddingi (Content-Based)": [],
                "D. Hybryda (ALS + Embeddingi)": [],
            }

            for uid, pos_list in cold_user_pos.items():
                onboarding_items = pos_list[:n_val]
                seen_set = set(onboarding_items)
                # Stały zbiór testowy (pozytywne filmy od 11. wzwyż), aby krzywa NDCG(N) była w 100% porównywalna
                rel_set = set(pos_list[10:])

                # A. MostPopular
                recs_a = most_pop.recommend(uid, seen_set, k=k)
                metrics_acc["A. MostPopular"].append(
                    (
                        ndcg_at_k(recs_a, rel_set, k),
                        precision_at_k(recs_a, rel_set, k),
                        recall_at_k(recs_a, rel_set, k),
                    )
                )

                if n_val == 0:
                    # Przy N=0 modele B, C, D nie mają jeszcze wybranych filmów -> startują z poziomu bazowego (MostPopular)
                    for variant_key in [
                        "B. CF (ALS fold-in)",
                        "C. Embeddingi (Content-Based)",
                        "D. Hybryda (ALS + Embeddingi)",
                    ]:
                        metrics_acc[variant_key].append(metrics_acc["A. MostPopular"][-1])
                else:
                    # B. CF (ALS fold-in)
                    recs_b = als_model.recommend_cold_start(onboarding_items, seen_set, k=k)
                    metrics_acc["B. CF (ALS fold-in)"].append(
                        (
                            ndcg_at_k(recs_b, rel_set, k),
                            precision_at_k(recs_b, rel_set, k),
                            recall_at_k(recs_b, rel_set, k),
                        )
                    )

                    # C. Embeddingi (Content-Based)
                    recs_c = cb_model.recommend_cold_start(onboarding_items, seen_set, k=k)
                    metrics_acc["C. Embeddingi (Content-Based)"].append(
                        (
                            ndcg_at_k(recs_c, rel_set, k),
                            precision_at_k(recs_c, rel_set, k),
                            recall_at_k(recs_c, rel_set, k),
                        )
                    )

                    # D. Hybryda
                    recs_d = hybrid_cs.recommend_cold_start(onboarding_items, seen_set, k=k)
                    metrics_acc["D. Hybryda (ALS + Embeddingi)"].append(
                        (
                            ndcg_at_k(recs_d, rel_set, k),
                            precision_at_k(recs_d, rel_set, k),
                            recall_at_k(recs_d, rel_set, k),
                        )
                    )

            for variant_name, tuples_list in metrics_acc.items():
                arr = np.array(tuples_list, dtype=np.float64)
                onboarding_records.append(
                    {
                        "seed": seed,
                        "N": n_val,
                        "Wariant": variant_name,
                        "NDCG@10": float(arr[:, 0].mean()),
                        "Precision@10": float(arr[:, 1].mean()),
                        "Recall@10": float(arr[:, 2].mean()),
                    }
                )

    stage1_df = (
        pd.DataFrame(stage1_records)
        .groupby("Wariant", as_index=False)[["NDCG@10", "Precision@10", "Recall@10"]]
        .mean()
        .sort_values("NDCG@10", ascending=False)
    )

    onboarding_df = (
        pd.DataFrame(onboarding_records)
        .groupby(["N", "Wariant"], as_index=False)[["NDCG@10", "Precision@10", "Recall@10"]]
        .mean()
    )

    return stage1_df, onboarding_df


def select_onboarding_candidates(
    ratings: pd.DataFrame, movies: pd.DataFrame, n_candidates: int = 30
) -> pd.DataFrame:
    """
    Wybiera ~30 filmów do ekranu onboardingu zgodnie z kryteriami z założeń:
    - znane (wysoka liczba ocen, min. 400 ocen),
    - dzielące widzów (wysokie odchylenie standardowe ocen std_rating),
    - z różnych gatunków (zbalansowana reprezentacja gatunków).
    """
    stats = ratings.groupby("movieId")["rating"].agg(["count", "mean", "std"]).reset_index()
    popular = stats[stats["count"] >= 400].copy()
    # Wskaźnik polaryzacji łączący wariancję ocen z logarytmem popularności
    popular["polarization_score"] = popular["std"] * np.log10(popular["count"])

    merged = popular.merge(
        movies[["movieId", "title", "genres", "director", "poster_path"]],
        on="movieId",
        how="left",
    )
    merged["primary_genre"] = merged["genres"].apply(
        lambda g: g[0] if isinstance(g, (list, np.ndarray)) and len(g) > 0 else "Other"
    )
    merged = merged.sort_values("polarization_score", ascending=False)

    selected_rows: list[pd.Series] = []
    chosen_ids: set[int] = set()
    genre_counts: dict[str, int] = {}

    for _, row in merged.iterrows():
        g = str(row["primary_genre"])
        if genre_counts.get(g, 0) >= 4:
            continue
        selected_rows.append(row)
        chosen_ids.add(int(row["movieId"]))
        genre_counts[g] = genre_counts.get(g, 0) + 1
        if len(selected_rows) >= n_candidates:
            break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def create_plots(
    overall_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    onboarding_df: pd.DataFrame,
) -> None:
    """Generuje czytelne wykresy w wysokiej rozdzielczości do umieszczenia w pracy."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Wykres Cold Start: NDCG@10 w funkcji liczby wybranych filmów N
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
    styles = {
        "A. MostPopular": ("#7f8c8d", "--", "o"),
        "B. CF (ALS fold-in)": ("#2980b9", "-.", "s"),
        "C. Embeddingi (Content-Based)": ("#e67e22", ":", "^"),
        "D. Hybryda (ALS + Embeddingi)": ("#27ae60", "-", "D"),
    }
    for variant, (color, ls, marker) in styles.items():
        sub = onboarding_df[onboarding_df["Wariant"] == variant].sort_values("N")
        ax.plot(
            sub["N"],
            sub["NDCG@10"],
            label=variant,
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=2.4,
            markersize=7,
        )

    ax.set_title("Cold Start: Trafność rekomendacji (NDCG@10) w zależności od liczby filmów N", fontsize=12, fontweight="bold")
    ax.set_xlabel("Liczba wybranych filmów podczas onboardingu (N)", fontsize=11)
    ax.set_ylabel("NDCG@10", fontsize=11)
    ax.set_xticks([0, 1, 3, 5, 10])
    ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "cold_start_ndcg_vs_N.png")
    plt.close(fig)

    # 2. Wykres NDCG@10 w grupach aktywności użytkowników (1-5, 6-20, 21-100, >100)
    fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=180)
    buckets = ["1-5", "6-20", "21-100", ">100"]
    models_to_plot = [
        ("MostPopular", "#95a5a6"),
        ("Content-Based (Transformer)", "#e67e22"),
        ("CF - ALS (implicit)", "#2980b9"),
        ("Hybryda (dynamiczna α)", "#27ae60"),
    ]
    x = np.arange(len(buckets))
    width = 0.20

    for idx, (m_name, color) in enumerate(models_to_plot):
        sub = activity_df[activity_df["Model"] == m_name]
        vals = [
            float(sub[sub["Grupa aktywności"] == b]["NDCG@10"].iloc[0])
            for b in buckets
        ]
        offset = (idx - 1.5) * width
        bars = ax.bar(x + offset, vals, width=width, label=m_name, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    ax.set_title("Porównanie NDCG@10 w zależności od liczby ocen użytkownika w historii", fontsize=12, fontweight="bold")
    ax.set_xlabel("Grupa aktywności użytkownika (liczba ocen treningowych)", fontsize=11)
    ax.set_ylabel("NDCG@10", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b} ocen" for b in buckets])
    ax.legend(frameon=True, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "activity_groups_ndcg.png")
    plt.close(fig)

    # 3. Ogólne porównanie wszystkich badanych modeli (NDCG@10)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=180)
    ordered = overall_df.sort_values("NDCG@10", ascending=True)
    colors = ["#34495e" if "Hybryda" not in m else "#27ae60" for m in ordered["Model"]]
    bars = ax.barh(ordered["Model"], ordered["NDCG@10"], color=colors, height=0.6)
    for bar in bars:
        w = bar.get_width()
        ax.annotate(
            f" {w:.4f}",
            xy=(w, bar.get_y() + bar.get_height() / 2),
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlim(0, float(ordered["NDCG@10"].max()) * 1.15)
    ax.set_title("Ogólne porównanie jakości rekomendacji Top-10 (NDCG@10)", fontsize=12, fontweight="bold")
    ax.set_xlabel("NDCG@10", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "overall_models_ndcg.png")
    plt.close(fig)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Wczytywanie przetworzonych danych i embeddingów Transformer ...")
    ratings = pd.read_parquet(PROCESSED_DIR / "ratings.parquet")
    movies = pd.read_parquet(PROCESSED_DIR / "movies_enriched.parquet")
    embeddings = np.load(PROCESSED_DIR / "movie_embeddings.npy")
    movie_ids = np.load(PROCESSED_DIR / "movie_ids.npy")

    print("[2/5] Tworzenie podziału trening/test z grupami aktywności (1-5, 6-20, 21-100, >100) ...")
    split = create_main_split_with_activity_groups(ratings, test_ratio=0.2, seed=42)

    bucket_counts = pd.Series(list(split.user_buckets.values())).value_counts()
    print("  Rozkład użytkowników w grupach aktywności:")
    for b in ["1-5", "6-20", "21-100", ">100"]:
        print(f"    {b:>6} ocen: {bucket_counts.get(b, 0):,} użytkowników")

    print("[3/5] Trenowanie modeli: MostPopular, Content-Based, SVD, ALS oraz Hybrydy ...")
    most_pop = MostPopularRecommender().fit(split.train_df)
    cb_model = ContentBasedRecommender(embeddings, movie_ids).fit(split.train_df)
    svd_model = SVDRecommender(movie_ids, factors=64, epochs=15, random_state=42).fit(split.train_df)
    als_model = ALSRecommender(
        movie_ids, factors=64, regularization=0.05, alpha_conf=15.0, iterations=20, random_state=42
    ).fit(split.train_df)

    hybrid_fixed = HybridRecommender(
        cb_model=cb_model,
        als_model=als_model,
        user_train_counts=split.user_train_counts,
        k_param=20.0,
        fixed_alpha=0.5,
    )
    hybrid_dynamic = HybridRecommender(
        cb_model=cb_model,
        als_model=als_model,
        user_train_counts=split.user_train_counts,
        k_param=20.0,
        fixed_alpha=None,
    )

    # Ewaluacja RMSE / MAE dla predykcji ocen w skali 1-5 (SVD vs Global Mean)
    y_true = split.test_df["rating"].to_numpy(dtype=np.float32)
    y_pred_svd = svd_model.predict_batch(split.test_df)
    y_pred_mean = np.full_like(y_true, float(split.train_df["rating"].mean()))

    rating_metrics_df = pd.DataFrame(
        [
            {
                "Model": "Średnia globalna (Baseline)",
                "RMSE": rmse(y_true, y_pred_mean),
                "MAE": mae(y_true, y_pred_mean),
            },
            {
                "Model": "CF - SVD (faktoryzacja macierzy jawnych ocen)",
                "RMSE": rmse(y_true, y_pred_svd),
                "MAE": mae(y_true, y_pred_svd),
            },
        ]
    )
    rating_metrics_df.to_csv(OUTPUTS_DIR / "svd_rating_metrics.csv", index=False)

    # Ewaluacja rankingowa Top-10 wszystkich modeli
    all_models = {
        "MostPopular": most_pop,
        "CF - SVD (jawne oceny)": svd_model,
        "Content-Based (Transformer)": cb_model,
        "CF - ALS (implicit)": als_model,
        "Hybryda (stała α=0.5)": hybrid_fixed,
        "Hybryda (dynamiczna α)": hybrid_dynamic,
    }

    overall_rows: list[dict[str, object]] = []
    for m_name, m_obj in all_models.items():
        res = evaluate_recommendations(
            m_obj.recommend, split.test_relevant, split.train_seen, k=10
        )
        overall_rows.append(
            {
                "Model": m_name,
                "NDCG@10": res["NDCG@10"],
                "Precision@10": res["Precision@10"],
                "Recall@10": res["Recall@10"],
            }
        )
    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(OUTPUTS_DIR / "overall_ranking_metrics.csv", index=False)

    print("[4/5] Ewaluacja modeli w grupach aktywności użytkowników (1-5, 6-20, 21-100, >100) ...")
    activity_models = {
        "MostPopular": most_pop,
        "Content-Based (Transformer)": cb_model,
        "CF - ALS (implicit)": als_model,
        "Hybryda (stała α=0.5)": hybrid_fixed,
        "Hybryda (dynamiczna α)": hybrid_dynamic,
    }
    activity_df = evaluate_by_activity_groups(
        activity_models, split.test_relevant, split.train_seen, split.user_buckets, k=10
    )
    activity_df.to_csv(OUTPUTS_DIR / "activity_groups_metrics.csv", index=False)

    print("[5/5] Eksperyment Cold Start (Etap 1: N=0 oraz Etap 2: Onboarding dla N=0,1,3,5,10) ...")
    stage1_df, onboarding_df = run_cold_start_experiment(
        ratings=ratings,
        movies=movies,
        embeddings=embeddings,
        movie_ids=movie_ids,
        n_values=[0, 1, 3, 5, 10],
        seeds=[42, 101, 202],
    )
    stage1_df.to_csv(OUTPUTS_DIR / "cold_start_stage1_n0.csv", index=False)
    onboarding_df.to_csv(OUTPUTS_DIR / "cold_start_onboarding.csv", index=False)

    onboarding_movies_df = select_onboarding_candidates(ratings, movies, n_candidates=30)
    onboarding_movies_df.to_csv(OUTPUTS_DIR / "onboarding_30_movies.csv", index=False)

    create_plots(overall_df, activity_df, onboarding_df)

    # Podsumowanie w konsoli
    print("\n" + "=" * 72)
    print("1. PREDYKCJA OCEN GWIAZDKOWYCH (SVD - RMSE / MAE)")
    print("=" * 72)
    print(rating_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("2. OGÓLNE WYNIKI RANKINGOWE TOP-10 (WSZYSTKIE MODELE)")
    print("=" * 72)
    print(overall_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("3. WYNIKI NDCG@10 W GRUPACH AKTYWNOŚCI UŻYTKOWNIKÓW")
    print("=" * 72)
    pivot_act = activity_df.pivot(index="Grupa aktywności", columns="Model", values="NDCG@10")
    pivot_act = pivot_act.reindex(["1-5", "6-20", "21-100", ">100"])
    print(pivot_act.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("4. COLD START ETAP 1 (N = 0, ZERO INFORMACJI)")
    print("=" * 72)
    print(stage1_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "=" * 72)
    print("5. COLD START ETAP 2 – ONBOARDING (NDCG@10 W ZALEŻNOŚCI OD N)")
    print("=" * 72)
    pivot_cs = onboarding_df.pivot(index="N", columns="Wariant", values="NDCG@10")
    print(pivot_cs.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 72)
    print(f"\n[INFO] Wszystkie tabele i wykresy zapisano w: {OUTPUTS_DIR}\n")


if __name__ == "__main__":
    main()
