"""
Moduł podziału danych oraz metryk ewaluacyjnych:
- Metryki rankingowe: NDCG@K, Precision@K, Recall@K
- Metryki błędów oceny: RMSE, MAE
- Podziały danych:
  1. Podział chronologiczny z grupami aktywności użytkowników (1-5, 6-20, 21-100, >100 ocen)
  2. Podział do symulacji Cold Start (nowi użytkownicy usunięci z treningu CF,
     N in {0, 1, 3, 5, 10} pierwszych pozytywnych ocen jako onboarding)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


# ============================================================================
# METRYKI RANKINGOWE I PREDYKCJI OCEN
# ============================================================================

def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(rec_k):
        if item in relevant:
            dcg += 1.0 / math.log2(idx + 2)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_recommendations(
    recommend_fn: Callable[[int, set[int], int], list[int]],
    test_relevant: dict[int, set[int]],
    train_seen: dict[int, set[int]],
    k: int = 10,
) -> dict[str, float]:
    """
    Oblicza średnie NDCG@k, Precision@k i Recall@k po wszystkich użytkownikach testowych,
    którzy posiadają co najmniej jeden trafny (relevant, ocena >= 4) film w zbiorze testowym.
    """
    ndcgs: list[float] = []
    precs: list[float] = []
    recalls: list[float] = []

    for user_id, rel_items in test_relevant.items():
        if not rel_items:
            continue
        seen = train_seen.get(user_id, set())
        recs = recommend_fn(user_id, seen, k)
        ndcgs.append(ndcg_at_k(recs, rel_items, k))
        precs.append(precision_at_k(recs, rel_items, k))
        recalls.append(recall_at_k(recs, rel_items, k))

    return {
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"Precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "n_eval_users": len(ndcgs),
    }


# ============================================================================
# PODZIAŁY DANYCH
# ============================================================================

def get_activity_bucket(n_ratings: int) -> str:
    """Przypisuje użytkownika do grupy aktywności wg liczby ocen w zbiorze treningowym."""
    if n_ratings <= 5:
        return "1-5"
    if n_ratings <= 20:
        return "6-20"
    if n_ratings <= 100:
        return "21-100"
    return ">100"


@dataclass
class MainSplit:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_seen: dict[int, set[int]]
    test_relevant: dict[int, set[int]]
    user_train_counts: dict[int, int]
    user_buckets: dict[int, str]


def create_main_split_with_activity_groups(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    relevant_threshold: int = 4,
    seed: int = 42,
) -> MainSplit:
    """
    Tworzy chronologiczny podział ocen każdego użytkownika na część treningową i testową.
    Ponieważ w MovieLens 1M każdy użytkownik ma co najmniej 20 ocen ogółem,
    aby zbadać zachowanie modeli we wszystkich czterech przedziałach aktywności
    opisanych w założeniach pracy:
      - '1-5' ocen
      - '6-20' ocen
      - '21-100' ocen
      - '>100' ocen
    dla części użytkowników losowo przycinamy historię treningową do zadanej długości
    (zgodnie z punktem: 'Każdy użytkownik ma co najmniej 20 ocen, więc da się symulować
    nowych użytkowników, ukrywając im historię').
    """
    rng = np.random.default_rng(seed)

    # Sortowanie chronologiczne wewnątrz każdego użytkownika
    sorted_df = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)

    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    all_users = sorted_df["userId"].unique()
    shuffled_users = rng.permutation(all_users)

    # Przydzielamy po ~15% użytkowników do symulowanych krótkich historii '1-5' oraz '6-20',
    # a pozostałych 70% zostawiamy z ich naturalną pełną historią chronologiczną ('21-100' oraz '>100').
    n_total = len(shuffled_users)
    short_1_5_users = set(shuffled_users[: int(0.15 * n_total)])
    short_6_20_users = set(shuffled_users[int(0.15 * n_total) : int(0.30 * n_total)])

    for user_id, group in sorted_df.groupby("userId", sort=False):
        n = len(group)
        n_test = max(5, int(round(n * test_ratio)))
        n_train_full = n - n_test

        train_part = group.iloc[:n_train_full]
        test_part = group.iloc[n_train_full:]

        uid = int(user_id)
        if uid in short_1_5_users:
            # Zostawiamy od 1 do 5 ocen treningowych (preferując takie, gdzie jest choć jedna >= 4)
            target_len = int(rng.integers(1, 6))
            pos_part = train_part[train_part["rating"] >= relevant_threshold]
            if len(pos_part) >= target_len:
                train_part = pos_part.iloc[:target_len]
            else:
                train_part = train_part.iloc[:target_len]
        elif uid in short_6_20_users:
            target_len = int(rng.integers(6, 21))
            train_part = train_part.iloc[: min(len(train_part), target_len)]

        train_parts.append(train_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_seen: dict[int, set[int]] = (
        train_df.groupby("userId")["movieId"].apply(set).to_dict()
    )
    user_train_counts: dict[int, int] = (
        train_df.groupby("userId").size().to_dict()
    )
    user_buckets: dict[int, str] = {
        int(u): get_activity_bucket(int(cnt)) for u, cnt in user_train_counts.items()
    }

    test_pos = test_df[test_df["rating"] >= relevant_threshold]
    test_relevant: dict[int, set[int]] = (
        test_pos.groupby("userId")["movieId"].apply(set).to_dict()
    )

    return MainSplit(
        train_df=train_df,
        test_df=test_df,
        train_seen=train_seen,
        test_relevant=test_relevant,
        user_train_counts=user_train_counts,
        user_buckets=user_buckets,
    )
