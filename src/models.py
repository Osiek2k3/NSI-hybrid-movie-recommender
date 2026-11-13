"""
Implementacja wszystkich modeli rekomendacyjnych opisanych w założeniach projektu:
1. MostPopularRecommender – najczęściej oceniane filmy (model bazowy)
2. BayesianPopularRecommender – średnia ważona (bayesowska) łącząca średnią ocenę z liczbą ocen (Cold Start etap 1)
3. DiversePopularRecommender – popularne i zróżnicowane gatunkowo (Cold Start etap 1)
4. ContentBasedRecommender – profil użytkownika z embeddingów Transformer (all-MiniLM-L6-v2, 384 wym.) + podobieństwo kosinusowe
5. ALSRecommender – filtrowanie kolaboratywne (implicit ALS) pod ranking Top-10 + obsługa fold-in dla nowych użytkowników
6. SVDRecommender – faktoryzacja macierzy z biasami na jawnych ocenach 1–5 (pod RMSE/MAE, przewidywaną ocenę w wyjaśnieniach oraz porównanie z ALS)
7. HybridRecommender – łączenie ALS i Content-Based wagą dynamiczną alpha(n) = n / (n + k) lub stałą wagą alpha
"""

from __future__ import annotations

import os
from pathlib import Path

# Wyciszenie ostrzeżenia OpenMP z biblioteki implicit na Windows
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


def min_max_normalize(scores: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Normalizuje wyniki kandydatów do przedziału [0, 1] (dla uczciwego łączenia w hybrydzie)."""
    out = np.zeros_like(scores, dtype=np.float32)
    valid = scores if mask is None else scores[mask]
    if valid.size == 0:
        return out
    s_min = float(np.min(valid))
    s_max = float(np.max(valid))
    if s_max - s_min < 1e-8:
        return out
    if mask is None:
        return ((scores - s_min) / (s_max - s_min)).astype(np.float32)
    out[mask] = (scores[mask] - s_min) / (s_max - s_min)
    return out


# ============================================================================
# 1. MODELE BAZOWE I COLD START ETAP 1 (ZERO INFORMACJI)
# ============================================================================

class MostPopularRecommender:
    """Poleca filmy z największą liczbą ocen w zbiorze treningowym."""

    def __init__(self) -> None:
        self.ranked_movies: list[int] = []
        self.pop_scores: dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "MostPopularRecommender":
        counts = train_df.groupby("movieId").size().sort_values(ascending=False)
        self.ranked_movies = [int(m) for m in counts.index]
        max_c = float(counts.max()) if len(counts) > 0 else 1.0
        self.pop_scores = {int(m): float(c) / max_c for m, c in counts.items()}
        return self

    def recommend(self, user_id: int, seen_items: set[int], k: int = 10) -> list[int]:
        recs: list[int] = []
        for m in self.ranked_movies:
            if m not in seen_items:
                recs.append(m)
                if len(recs) >= k:
                    break
        return recs


class BayesianPopularRecommender:
    """
    Popularne i dobrze oceniane: sortowanie po średniej ważonej (średnia bayesowska):
      WR = (v / (v + m)) * R + (m / (v + m)) * C
    Dzięki temu film z 5.0 i trzema ocenami nie wyprzedzi filmu z 4.4 i dwoma tysiącami ocen.
    """

    def __init__(self, m_prior: float = 200.0) -> None:
        self.m_prior = m_prior
        self.ranked_movies: list[int] = []
        self.movie_scores: dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "BayesianPopularRecommender":
        global_mean = float(train_df["rating"].mean())
        stats = train_df.groupby("movieId")["rating"].agg(["count", "mean"])
        v = stats["count"].astype(float)
        r = stats["mean"].astype(float)
        m = self.m_prior
        stats["bayesian_score"] = (v / (v + m)) * r + (m / (v + m)) * global_mean
        stats = stats.sort_values("bayesian_score", ascending=False)
        self.ranked_movies = [int(mid) for mid in stats.index]
        self.movie_scores = {
            int(mid): float(sc) for mid, sc in stats["bayesian_score"].items()
        }
        return self

    def recommend(self, user_id: int, seen_items: set[int], k: int = 10) -> list[int]:
        recs: list[int] = []
        for m in self.ranked_movies:
            if m not in seen_items:
                recs.append(m)
                if len(recs) >= k:
                    break
        return recs


class DiversePopularRecommender:
    """
    Popularne i zróżnicowane (Cold Start Etap 1):
    Wybiera cyklicznie (round-robin) najlepsze popularne filmy z głównych gatunków,
    aby lista Top-10 pokrywała różne gusta zamiast 10 podobnych hitów.
    """

    MAIN_GENRES = [
        "Drama",
        "Comedy",
        "Action",
        "Sci-Fi",
        "Thriller",
        "Romance",
        "Adventure",
        "Crime",
        "Animation",
        "Horror",
    ]

    def __init__(self, min_votes: int = 300) -> None:
        self.min_votes = min_votes
        self.genre_queues: dict[str, list[int]] = {}
        self.fallback_ranked: list[int] = []

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> "DiversePopularRecommender":
        # Używamy popularności wśród wysokich ocen (liczba ocen >= 4)
        pos_df = train_df[train_df["rating"] >= 4]
        pos_counts = pos_df.groupby("movieId").size().sort_values(ascending=False)
        self.fallback_ranked = [int(m) for m in pos_counts.index]

        movie_genres: dict[int, list[str]] = {}
        for _, row in movies_df.iterrows():
            g_val = row.get("genres")
            if isinstance(g_val, (list, np.ndarray)):
                g_list = [str(x) for x in g_val]
            else:
                g_list = [x for x in str(g_val).split("|") if x]
            movie_genres[int(row["movieId"])] = g_list

        self.genre_queues = {g: [] for g in self.MAIN_GENRES}
        for mid in self.fallback_ranked:
            for g in movie_genres.get(mid, []):
                if g in self.genre_queues:
                    self.genre_queues[g].append(mid)

        return self

    def recommend(self, user_id: int, seen_items: set[int], k: int = 10) -> list[int]:
        recs: list[int] = []
        chosen: set[int] = set(seen_items)

        # Round-robin po głównych gatunkach
        while len(recs) < k:
            added_any = False
            for g in self.MAIN_GENRES:
                for mid in self.genre_queues.get(g, []):
                    if mid not in chosen:
                        recs.append(mid)
                        chosen.add(mid)
                        added_any = True
                        break
                if len(recs) >= k:
                    break
            if not added_any:
                break

        for mid in self.fallback_ranked:
            if len(recs) >= k:
                break
            if mid not in chosen:
                recs.append(mid)
                chosen.add(mid)

        return recs


# ============================================================================
# 2. CONTENT-BASED (EMBEDDINGI TRANSFORMER)
# ============================================================================

class ContentBasedRecommender:
    """
    Filtrowanie oparte na treści:
    - Każdy film reprezentowany jest wektorem 384 liczb z modelu all-MiniLM-L6-v2.
    - Profil użytkownika to średnia ważona embeddingów filmów ocenionych wysoko (>= 4).
    - Podobieństwo kosinusowe (Transformer) łączone jest z łagodnym priorem popularności/jakości
      filmu ze zbioru treningowego, aby spośród filmów o podobnej fabule (np. 'Toy Story 2'
      kontra niszowy film o zabawkach z kilkoma ocenami) preferować tytuły znane i cenione.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        movie_ids: np.ndarray,
        pop_prior_weight: float = 0.35,
    ) -> None:
        self.embeddings = embeddings.astype(np.float32)
        self.movie_ids = movie_ids.astype(np.int32)
        self.pop_prior_weight = pop_prior_weight
        self.id_to_idx: dict[int, int] = {
            int(mid): idx for idx, mid in enumerate(self.movie_ids)
        }
        self.user_profiles: dict[int, np.ndarray] = {}
        self.item_quality_prior: np.ndarray = np.ones(len(self.movie_ids), dtype=np.float32)

    def fit(self, train_df: pd.DataFrame) -> "ContentBasedRecommender":
        self.user_profiles.clear()

        # Wyznaczenie prioru popularności/jakości filmów z danych treningowych (w skali [0, 1])
        pos_train = train_df[train_df["rating"] >= 4]
        counts = pos_train.groupby("movieId").size()
        prior = np.zeros(len(self.movie_ids), dtype=np.float32)
        for mid, cnt in counts.items():
            idx = self.id_to_idx.get(int(mid))
            if idx is not None:
                prior[idx] = np.log1p(float(cnt))
        if prior.max() > 0:
            prior = prior / prior.max()
        self.item_quality_prior = prior

        # Budujemy profil z filmów ocenionych pozytywnie (>= 4), z wagą (rating - 2.5)
        for user_id, group in train_df.groupby("userId", sort=False):
            pos = group[group["rating"] >= 4]
            use_group = pos if len(pos) > 0 else group

            indices: list[int] = []
            weights: list[float] = []
            for mid, rating in zip(use_group["movieId"], use_group["rating"]):
                idx = self.id_to_idx.get(int(mid))
                if idx is not None:
                    indices.append(idx)
                    weights.append(max(0.5, float(rating) - 2.5))

            if indices:
                vecs = self.embeddings[indices]
                w = np.array(weights, dtype=np.float32)[:, None]
                profile = (vecs * w).sum(axis=0)
                norm = float(np.linalg.norm(profile))
                if norm > 1e-8:
                    profile = profile / norm
                self.user_profiles[int(user_id)] = profile.astype(np.float32)

        return self

    def build_profile_from_items(self, liked_movie_ids: list[int]) -> np.ndarray | None:
        """Buduje wektor profilu dla nowego użytkownika na podstawie wybranych filmów (onboarding)."""
        indices = [
            self.id_to_idx[int(mid)]
            for mid in liked_movie_ids
            if int(mid) in self.id_to_idx
        ]
        if not indices:
            return None
        profile = self.embeddings[indices].mean(axis=0)
        norm = float(np.linalg.norm(profile))
        if norm > 1e-8:
            profile = profile / norm
        return profile.astype(np.float32)

    def score_all_items(self, user_id: int) -> np.ndarray:
        """Zwraca wynik dopasowania treściowego dla wszystkich filmów z katalogu."""
        profile = self.user_profiles.get(int(user_id))
        return self.score_all_from_profile(profile)

    def score_all_from_profile(self, profile: np.ndarray | None) -> np.ndarray:
        if profile is None:
            return np.zeros(len(self.movie_ids), dtype=np.float32)
        cos_sim = np.clip(self.embeddings @ profile, 0.0, 1.0)
        if self.pop_prior_weight <= 0.0:
            return cos_sim.astype(np.float32)
        # Iloczyn podobieństwa semantycznego Transformer oraz prioru rozpoznawalności filmu
        return (
            cos_sim * ((1.0 - self.pop_prior_weight) + self.pop_prior_weight * (self.item_quality_prior ** 1.5))
        ).astype(np.float32)

    def recommend(self, user_id: int, seen_items: set[int], k: int = 10) -> list[int]:
        scores = self.score_all_items(user_id)
        return self._top_k_from_scores(scores, seen_items, k)

    def recommend_cold_start(
        self, liked_movie_ids: list[int], seen_items: set[int], k: int = 10
    ) -> list[int]:
        profile = self.build_profile_from_items(liked_movie_ids)
        scores = self.score_all_from_profile(profile)
        return self._top_k_from_scores(scores, seen_items, k)

    def _top_k_from_scores(
        self, scores: np.ndarray, seen_items: set[int], k: int
    ) -> list[int]:
        scores_copy = scores.copy()
        for mid in seen_items:
            idx = self.id_to_idx.get(int(mid))
            if idx is not None:
                scores_copy[idx] = -1e9
        top_idx = np.argpartition(-scores_copy, kth=min(k, len(scores_copy) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores_copy[top_idx])]
        return [int(self.movie_ids[i]) for i in top_idx]


# ============================================================================
# 3. FILTROWANIE KOLABORATYWNE – ALS (IMPLICIT) + FOLD-IN
# ============================================================================

class ALSRecommender:
    """
    Filtrowanie kolaboratywne w wariancie ALS (biblioteka implicit) uczone pod ranking Top-10.
    Obsługuje również szybkie wyznaczanie wektora nowego użytkownika metodą fold-in
    (recalculate_user=True) bez ponownego uczenia modelu.
    """

    def __init__(
        self,
        movie_ids: np.ndarray,
        factors: int = 64,
        regularization: float = 0.05,
        alpha_conf: float = 15.0,
        iterations: int = 20,
        random_state: int = 42,
    ) -> None:
        self.movie_ids = movie_ids.astype(np.int32)
        self.id_to_idx: dict[int, int] = {
            int(mid): idx for idx, mid in enumerate(self.movie_ids)
        }
        self.factors = factors
        self.regularization = regularization
        self.alpha_conf = alpha_conf
        self.iterations = iterations
        self.random_state = random_state

        self.model: AlternatingLeastSquares | None = None
        self.user_to_idx: dict[int, int] = {}
        self.idx_to_user: dict[int, int] = {}
        self.user_items_csr: sp.csr_matrix | None = None

    def fit(self, train_df: pd.DataFrame) -> "ALSRecommender":
        unique_users = np.sort(train_df["userId"].unique())
        self.user_to_idx = {int(u): idx for idx, u in enumerate(unique_users)}
        self.idx_to_user = {idx: int(u) for idx, u in enumerate(unique_users)}

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        # Budujemy macierz implicit z pozytywnych interakcji (oceny >= 4)
        pos_df = train_df[train_df["rating"] >= 4]
        for u, m, r in zip(pos_df["userId"], pos_df["movieId"], pos_df["rating"]):
            u_idx = self.user_to_idx.get(int(u))
            m_idx = self.id_to_idx.get(int(m))
            if u_idx is not None and m_idx is not None:
                weight = 1.0 + self.alpha_conf * (float(r) - 3.0)
                rows.append(u_idx)
                cols.append(m_idx)
                vals.append(weight)

        n_users = len(unique_users)
        n_items = len(self.movie_ids)
        self.user_items_csr = sp.csr_matrix(
            (np.array(vals, dtype=np.float32), (rows, cols)),
            shape=(n_users, n_items),
        )

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
        )
        self.model.fit(self.user_items_csr, show_progress=False)
        return self

    def score_all_items(self, user_id: int) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(self.movie_ids), dtype=np.float32)
        u_idx = self.user_to_idx.get(int(user_id))
        if u_idx is None:
            return np.zeros(len(self.movie_ids), dtype=np.float32)
        user_vec = self.model.user_factors[u_idx]
        return (self.model.item_factors @ user_vec).astype(np.float32)

    def fold_in_scores(self, liked_movie_ids: list[int]) -> np.ndarray:
        """
        Oblicza wektor nowego użytkownika (fold-in) z wektorów ALS wybranych filmów
        bez ponownego uczenia modelu i zwraca przewidywane wyniki dla wszystkich filmów.
        """
        if self.model is None or not liked_movie_ids:
            return np.zeros(len(self.movie_ids), dtype=np.float32)

        cols = [
            self.id_to_idx[int(m)]
            for m in liked_movie_ids
            if int(m) in self.id_to_idx
        ]
        if not cols:
            return np.zeros(len(self.movie_ids), dtype=np.float32)

        vals = np.full(len(cols), 1.0 + self.alpha_conf, dtype=np.float32)
        rows = np.zeros(len(cols), dtype=np.int32)
        user_row = sp.csr_matrix(
            (vals, (rows, cols)), shape=(1, len(self.movie_ids))
        )
        user_vec = self.model.recalculate_user(0, user_row)
        if user_vec.ndim == 2:
            user_vec = user_vec[0]
        return (self.model.item_factors @ user_vec).astype(np.float32)

    def recommend(self, user_id: int, seen_items: set[int], k: int = 10) -> list[int]:
        scores = self.score_all_items(user_id)
        return self._top_k_from_scores(scores, seen_items, k)

    def recommend_cold_start(
        self, liked_movie_ids: list[int], seen_items: set[int], k: int = 10
    ) -> list[int]:
        scores = self.fold_in_scores(liked_movie_ids)
        return self._top_k_from_scores(scores, seen_items, k)

    def _top_k_from_scores(
        self, scores: np.ndarray, seen_items: set[int], k: int
    ) -> list[int]:
        scores_copy = scores.copy()
        for mid in seen_items:
            idx = self.id_to_idx.get(int(mid))
            if idx is not None:
                scores_copy[idx] = -1e9
        top_idx = np.argpartition(-scores_copy, kth=min(k, len(scores_copy) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores_copy[top_idx])]
        return [int(self.movie_ids[i]) for i in top_idx]


# ============================================================================
# 4. FILTROWANIE KOLABORATYWNE – SVD (JAWNE OCENY 1-5 GWIAZDEK)
# ============================================================================

class SVDRecommender:
    """
    Klasyczna faktoryzacja macierzy SVD z biasami (wzorowana na algorytmie Simon Funk / Surprise SVD):
      r_hat(u, i) = mu + b_u + b_i + p_u^T q_i
    Uczona wyłącznie na wystawionych ocenach (minimalizacja błędu kwadratowego RMSE).
    Służy w projekcie do:
      1. Obliczenia metryk dokładności predykcji ocen (RMSE, MAE)
      2. Dostarczenia przewidywanej liczby gwiazdek (1-5) do wyjaśnień LLM
      3. Pokazania na wynikach rankingowych (NDCG@10), dlaczego niski błąd RMSE
         nie oznacza dobrej listy Top-10 w porównaniu z ALS.
    """

    def __init__(
        self,
        movie_ids: np.ndarray,
        factors: int = 64,
        lr: float = 0.007,
        reg: float = 0.02,
        epochs: int = 15,
        random_state: int = 42,
    ) -> None:
        self.movie_ids = movie_ids.astype(np.int32)
        self.id_to_idx: dict[int, int] = {
            int(mid): idx for idx, mid in enumerate(self.movie_ids)
        }
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.random_state = random_state

        self.global_mean: float = 3.5
        self.user_to_idx: dict[int, int] = {}
        self.b_u: np.ndarray = np.array([], dtype=np.float32)
        self.b_i: np.ndarray = np.array([], dtype=np.float32)
        self.P: np.ndarray = np.array([], dtype=np.float32)
        self.Q: np.ndarray = np.array([], dtype=np.float32)

    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        rng = np.random.default_rng(self.random_state)
        unique_users = np.sort(train_df["userId"].unique())
        self.user_to_idx = {int(u): idx for idx, u in enumerate(unique_users)}

        n_users = len(unique_users)
        n_items = len(self.movie_ids)
        self.global_mean = float(train_df["rating"].mean())

        self.b_u = np.zeros(n_users, dtype=np.float32)
        self.b_i = np.zeros(n_items, dtype=np.float32)
        self.P = rng.normal(0.0, 0.05, size=(n_users, self.factors)).astype(np.float32)
        self.Q = rng.normal(0.0, 0.05, size=(n_items, self.factors)).astype(np.float32)

        u_indices = np.array(
            [self.user_to_idx[int(u)] for u in train_df["userId"]], dtype=np.int32
        )
        i_indices = np.array(
            [self.id_to_idx[int(m)] for m in train_df["movieId"]], dtype=np.int32
        )
        ratings = train_df["rating"].to_numpy(dtype=np.float32)

        # Szybkie mini-batch SGD ze średnim gradientem na użytkownika/film w batchu
        batch_size = 512
        n_samples = len(ratings)

        for epoch in range(self.epochs):
            perm = rng.permutation(n_samples)
            u_shuf = u_indices[perm]
            i_shuf = i_indices[perm]
            r_shuf = ratings[perm]

            lr_epoch = self.lr * (0.95 ** epoch)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                ub = u_shuf[start:end]
                ib = i_shuf[start:end]
                rb = r_shuf[start:end]

                pred = (
                    self.global_mean
                    + self.b_u[ub]
                    + self.b_i[ib]
                    + np.sum(self.P[ub] * self.Q[ib], axis=1)
                )
                err = rb - pred

                np.add.at(self.b_u, ub, lr_epoch * (err - self.reg * self.b_u[ub]))
                np.add.at(self.b_i, ib, lr_epoch * (err - self.reg * self.b_i[ib]))

                p_old = self.P[ub].copy()
                q_old = self.Q[ib].copy()
                np.add.at(
                    self.P,
                    ub,
                    lr_epoch * (err[:, None] * q_old - self.reg * p_old),
                )
                np.add.at(
                    self.Q,
                    ib,
                    lr_epoch * (err[:, None] * p_old - self.reg * q_old),
                )

        return self

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        u_idx = self.user_to_idx.get(int(user_id))
        i_idx = self.id_to_idx.get(int(movie_id))
        pred = self.global_mean
        if u_idx is not None:
            pred += float(self.b_u[u_idx])
        if i_idx is not None:
            pred += float(self.b_i[i_idx])
        if u_idx is not None and i_idx is not None:
            pred += float(np.dot(self.P[u_idx], self.Q[i_idx]))
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = np.empty(len(test_df), dtype=np.float32)
        for idx, (u, m) in enumerate(zip(test_df["userId"], test_df["movieId"])):
            preds[idx] = self.predict_rating(int(u), int(m))
        return preds

    def recommend(self, user_id: int, seen_items: set[int], k: int = 10) -> list[int]:
        u_idx = self.user_to_idx.get(int(user_id))
        if u_idx is None:
            scores = self.global_mean + self.b_i
        else:
            scores = (
                self.global_mean
                + self.b_u[u_idx]
                + self.b_i
                + (self.Q @ self.P[u_idx])
            )
        scores_copy = scores.copy()
        for mid in seen_items:
            idx = self.id_to_idx.get(int(mid))
            if idx is not None:
                scores_copy[idx] = -1e9
        top_idx = np.argpartition(-scores_copy, kth=min(k, len(scores_copy) - 1))[:k]
        top_idx = top_idx[np.argsort(-scores_copy[top_idx])]
        return [int(self.movie_ids[i]) for i in top_idx]


# ============================================================================
# 5. HYBRYDA (CONTENT-BASED + ALS Z WAGĄ DYNAMICZNĄ LUB STAŁĄ)
# ============================================================================

class HybridRecommender:
    """
    Model hybrydowy łączący wyniki Content-Based (embeddingi Transformer)
    oraz Filtrowania Kolaboratywnego (ALS):
      score(u, i) = alpha(n_u) * score_CF(u, i) + (1 - alpha(n_u)) * score_CB(u, i)
    gdzie:
      alpha(n_u) = n_u / (n_u + k_param)
    (dla k_param = 20: n=0 -> 0%, n=5 -> 20%, n=20 -> 50%, n=200 -> 91%).
    """

    def __init__(
        self,
        cb_model: ContentBasedRecommender,
        als_model: ALSRecommender,
        user_train_counts: dict[int, int],
        k_param: float = 20.0,
        fixed_alpha: float | None = None,
    ) -> None:
        self.cb_model = cb_model
        self.als_model = als_model
        self.user_train_counts = user_train_counts
        self.k_param = k_param
        self.fixed_alpha = fixed_alpha
        self.movie_ids = cb_model.movie_ids
        self.id_to_idx = cb_model.id_to_idx

    def compute_alpha(self, n_ratings: int) -> float:
        if self.fixed_alpha is not None:
            return float(self.fixed_alpha)
        if n_ratings <= 0:
            return 0.0
        return float(n_ratings) / (float(n_ratings) + self.k_param)

    def recommend(self, user_id: int, seen_items: set[int], k: int = 10) -> list[int]:
        n_ratings = self.user_train_counts.get(int(user_id), 0)
        alpha = self.compute_alpha(n_ratings)

        cb_raw = self.cb_model.score_all_items(user_id)
        cf_raw = self.als_model.score_all_items(user_id)

        return self._combine_and_rank(cb_raw, cf_raw, alpha, seen_items, k)

    def recommend_cold_start(
        self, liked_movie_ids: list[int], seen_items: set[int], k: int = 10
    ) -> list[int]:
        n_ratings = len(liked_movie_ids)
        alpha = self.compute_alpha(n_ratings)

        profile = self.cb_model.build_profile_from_items(liked_movie_ids)
        cb_raw = self.cb_model.score_all_from_profile(profile)
        cf_raw = self.als_model.fold_in_scores(liked_movie_ids)

        return self._combine_and_rank(cb_raw, cf_raw, alpha, seen_items, k)

    def _combine_and_rank(
        self,
        cb_raw: np.ndarray,
        cf_raw: np.ndarray,
        alpha: float,
        seen_items: set[int],
        k: int,
    ) -> list[int]:
        mask = np.ones(len(self.movie_ids), dtype=bool)
        for mid in seen_items:
            idx = self.id_to_idx.get(int(mid))
            if idx is not None:
                mask[idx] = False

        cb_norm = min_max_normalize(cb_raw, mask)
        cf_norm = min_max_normalize(cf_raw, mask)

        combined = alpha * cf_norm + (1.0 - alpha) * cb_norm
        combined[~mask] = -1e9

        top_idx = np.argpartition(-combined, kth=min(k, len(combined) - 1))[:k]
        top_idx = top_idx[np.argsort(-combined[top_idx])]
        return [int(self.movie_ids[i]) for i in top_idx]
