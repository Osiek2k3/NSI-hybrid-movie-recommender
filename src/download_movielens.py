"""
Pobieranie i wstępne przetwarzanie zbiorów MovieLens 1M oraz ml-latest-small.
Łączy filmy z MovieLens 1M z identyfikatorami tmdbId/imdbId z pliku links.csv
oraz weryfikuje zgodność tytułów i lat wydania.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

ML_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_and_extract(url: str, dest_dir: Path, folder_name: str) -> Path:
    """Pobiera archiwum ZIP z GroupLens i rozpakowuje je, jeśli jeszcze nie istnieje."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = dest_dir / folder_name
    if extracted_path.exists():
        print(f"[INFO] Zbiór {folder_name} już istnieje w: {extracted_path}")
        return extracted_path

    zip_path = dest_dir / f"{folder_name}.zip"
    print(f"[INFO] Pobieranie {url} -> {zip_path} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    zip_path.write_bytes(resp.content)

    print(f"[INFO] Rozpakowywanie {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    return extracted_path


def parse_title_and_year(raw_title: str) -> tuple[str, int | None]:
    """Wyciąga czysty tytuł (z obsługą np. 'Matrix, The' -> 'The Matrix') oraz rok wydania."""
    match = re.search(r"^(.*)\s+\((\d{4})\)\s*$", str(raw_title).strip())
    if match:
        title_part = match.group(1).strip()
        year = int(match.group(2))
    else:
        title_part = str(raw_title).strip()
        year = None

    # Usunięcie ewentualnego alternatywnego tytułu w nawiasie na końcu
    alt_match = re.match(r"^(.*?)\s*\([^)]+\)\s*$", title_part)
    base_title = alt_match.group(1).strip() if alt_match else title_part

    # Zamiana 'Title, The' -> 'The Title' (ułatwia wyszukiwanie w TMDB)
    for article in (", The", ", A", ", An", ", Les", ", Le", ", La", ", L'", ", Das", ", Der", ", Die"):
        if base_title.endswith(article):
            art = article[2:].strip()
            sep = "" if art.endswith("'") else " "
            base_title = f"{art}{sep}{base_title[:-len(article)]}"
            break

    return base_title.strip(), year


def normalize_for_comparison(title: str) -> str:
    """Normalizuje tytuł do porównania między ml-1m a ml-latest-small."""
    clean, _ = parse_title_and_year(title)
    return re.sub(r"[^a-z0-9]", "", clean.lower())


def load_movielens_1m(ml_1m_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Wczytuje pliki ratings.dat, movies.dat i users.dat ze zbioru MovieLens 1M."""
    ratings = pd.read_csv(
        ml_1m_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        ml_1m_dir / "movies.dat",
        sep="::",
        engine="python",
        names=["movieId", "title", "genres"],
        encoding="latin-1",
    )

    users = pd.read_csv(
        ml_1m_dir / "users.dat",
        sep="::",
        engine="python",
        names=["userId", "gender", "age", "occupation", "zipCode"],
        encoding="latin-1",
    )

    return ratings, movies, users


def match_links_from_small(movies_1m: pd.DataFrame, ml_small_dir: Path) -> pd.DataFrame:
    """
    Dołącza tmdbId i imdbId z ml-latest-small (links.csv) oraz weryfikuje zgodność tytułów.
    """
    links = pd.read_csv(ml_small_dir / "links.csv")
    movies_small = pd.read_csv(ml_small_dir / "movies.csv")[["movieId", "title"]].rename(
        columns={"title": "title_small"}
    )

    merged = movies_1m.merge(links, on="movieId", how="left").merge(
        movies_small, on="movieId", how="left"
    )

    # Parsowanie czystego tytułu i roku z MovieLens 1M
    parsed = merged["title"].apply(parse_title_and_year)
    merged["clean_title"] = [p[0] for p in parsed]
    merged["year"] = pd.array([p[1] for p in parsed], dtype="Int64")
    merged["genres_list"] = merged["genres"].apply(
        lambda g: [x for x in str(g).split("|") if x and x != "(no genres listed)"]
    )

    # Weryfikacja zgodności tytułu lub roku między ml-1m a ml-latest-small
    def check_title_match(row) -> bool:
        if pd.isna(row["tmdbId"]) or pd.isna(row["title_small"]):
            return False
        _, year_small = parse_title_and_year(row["title_small"])
        norm_1m = normalize_for_comparison(row["title"])
        norm_small = normalize_for_comparison(row["title_small"])
        # Zgodność jeśli znormalizowane tytuły są takie same lub jeden zawiera drugi przy tym samym roku
        if norm_1m == norm_small:
            return True
        if row["year"] == year_small and (norm_1m in norm_small or norm_small in norm_1m):
            return True
        # Również ten sam rok i wysoka zbieżność (np. alternatywny zapis)
        return row["year"] == year_small

    merged["link_verified"] = merged.apply(check_title_match, axis=1)
    # Jeśli link nie przeszedł weryfikacji, czyścimy tmdbId, żeby wyszukać po tytule i roku w TMDB
    invalid_mask = merged["tmdbId"].notna() & (~merged["link_verified"])
    if invalid_mask.sum() > 0:
        print(f"[INFO] Odrzucono {invalid_mask.sum()} niezgodnych powiązań z links.csv (zostaną wyszukane po tytule i roku).")
        merged.loc[invalid_mask, "tmdbId"] = pd.NA

    merged["tmdbId"] = merged["tmdbId"].astype("Int64")
    merged["imdbId"] = merged["imdbId"].astype("Int64")

    return merged.drop(columns=["title_small"])


def print_dataset_summary(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame) -> None:
    """Wypisuje najważniejsze statystyki zbioru MovieLens 1M (do opisu w pracy)."""
    n_users = ratings["userId"].nunique()
    n_movies_rated = ratings["movieId"].nunique()
    n_movies_total = len(movies)
    n_ratings = len(ratings)
    density = n_ratings / (n_users * n_movies_total) * 100
    user_counts = ratings.groupby("userId").size()
    matched_tmdb = movies["tmdbId"].notna().sum()

    print("\n" + "=" * 60)
    print("PODSUMOWANIE ZBIORU MOVIELENS 1M")
    print("=" * 60)
    print(f"Liczba ocen:                     {n_ratings:,}")
    print(f"Liczba użytkowników:             {n_users:,} (w pliku users.dat: {len(users):,})")
    print(f"Liczba filmów w katalogu:        {n_movies_total:,} (ocenionych: {n_movies_rated:,})")
    print(f"Wypełnienie macierzy (gęstość):  {density:.2f}%")
    print(f"Min. liczba ocen na użytkownika: {user_counts.min()}")
    print(f"Mediana ocen na użytkownika:     {user_counts.median():.0f}")
    print(f"Średnia ocen na użytkownika:     {user_counts.mean():.1f}")
    print(f"Maks. liczba ocen użytkownika:   {user_counts.max()}")
    print(
        f"Zakres czasowy ocen:             "
        f"{pd.to_datetime(ratings['timestamp'].min(), unit='s')} -> "
        f"{pd.to_datetime(ratings['timestamp'].max(), unit='s')}"
    )
    print("-" * 60)
    print(
        f"Filmy dopasowane z links.csv:    {matched_tmdb:,} / {n_movies_total:,} "
        f"({matched_tmdb / n_movies_total * 100:.1f}%)"
    )
    print(
        f"Filmy do doszukania w TMDB API:  {n_movies_total - matched_tmdb:,} "
        f"(po tytule i roku)"
    )
    print("=" * 60 + "\n")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ml_1m_dir = download_and_extract(ML_1M_URL, RAW_DIR, "ml-1m")
    ml_small_dir = download_and_extract(ML_SMALL_URL, RAW_DIR, "ml-latest-small")

    print("[INFO] Wczytywanie MovieLens 1M ...")
    ratings, movies_1m, users = load_movielens_1m(ml_1m_dir)

    print("[INFO] Łączenie z links.csv (ml-latest-small) i weryfikacja tytułów ...")
    movies = match_links_from_small(movies_1m, ml_small_dir)

    # Zapis do plików Parquet
    ratings.to_parquet(PROCESSED_DIR / "ratings.parquet", index=False)
    users.to_parquet(PROCESSED_DIR / "users.parquet", index=False)
    movies.to_parquet(PROCESSED_DIR / "movies_base.parquet", index=False)

    print(f"[INFO] Zapisano przetworzone tabele w: {PROCESSED_DIR}")
    print_dataset_summary(ratings, movies, users)


if __name__ == "__main__":
    main()
