"""
Pobieranie metadanych filmowych z TMDB API (overview, keywords, genres, cast, director, poster_path)
dla zbioru MovieLens 1M i budowa tabeli data/processed/movies_enriched.parquet.

Zgodnie z założeniami pracy:
1. Dla filmów bez tmdbId w links.csv wyszukuje film w TMDB po tytule i roku (GET /3/search/movie).
2. Dla każdego tmdbId wysyła jedno zapytanie:
   GET /3/movie/{tmdbId}?append_to_response=credits,keywords&language=en-US
3. Zapisuje surowe odpowiedzi w pamięci podręcznej (data/raw/tmdb_cache.jsonl), aby w razie przerwania
   wznawiać bez ponownego odpytywania API.
4. Tworzy plik data/processed/movies_enriched.parquet oraz wypisuje statystykę pokrycia opisów.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CACHE_PATH = ROOT_DIR / "data" / "raw" / "tmdb_cache.jsonl"
TMDB_BASE_URL = "https://api.themoviedb.org/3"


class TMDBClient:
    """Klient HTTP dla TMDB API v3 (obsługuje zarówno API Key jak i Bearer Read Access Token)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=24, pool_maxsize=24)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        # Jeśli klucz jest długim tokenem JWT (v4 Read Access Token), używamy nagłówka Bearer
        if len(self.api_key) > 60 or "." in self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                }
            )
            self.use_query_param = False
        else:
            self.session.headers.update({"Accept": "application/json"})
            self.use_query_param = True

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        params = dict(params or {})
        if self.use_query_param:
            params["api_key"] = self.api_key

        url = f"{TMDB_BASE_URL}{endpoint}"
        for attempt in range(4):
            try:
                resp = self.session.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 404:
                    return None
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2.0))
                    time.sleep(retry_after + 0.5)
                    continue
                if resp.status_code == 401:
                    raise RuntimeError(
                        "Błąd autoryzacji TMDB API (401 Unauthorized). "
                        "Sprawdź poprawność klucza TMDB_API_KEY w pliku .env."
                    )
            except requests.RequestException:
                time.sleep(1.0 * (attempt + 1))
        return None

    def search_movie(self, title: str, year: int | None) -> int | None:
        """Szuka filmu w TMDB po tytule i roku (z tolerancją ±1 rok jako fallback)."""
        params: dict[str, Any] = {"query": title, "language": "en-US", "include_adult": "false"}
        if year is not None and not pd.isna(year):
            params["year"] = int(year)

        data = self._get("/search/movie", params=params)
        results = (data or {}).get("results", [])
        if results:
            return int(results[0]["id"])

        # Fallback: wyszukiwanie bez sztywnego filtra roku, ale z wyborem najbliższego roku (±2 lata)
        if year is not None and not pd.isna(year):
            params.pop("year", None)
            data = self._get("/search/movie", params=params)
            results = (data or {}).get("results", [])
            for item in results:
                rel_date = str(item.get("release_date") or "")
                if len(rel_date) >= 4 and rel_date[:4].isdigit():
                    rel_year = int(rel_date[:4])
                    if abs(rel_year - int(year)) <= 2:
                        return int(item["id"])
            if results:
                return int(results[0]["id"])

        return None

    def fetch_movie_details(self, tmdb_id: int) -> dict[str, Any] | None:
        """Pobiera szczegóły filmu wraz z obsadą i słowami kluczowymi w jednym zapytaniu."""
        return self._get(
            f"/movie/{int(tmdb_id)}",
            params={"append_to_response": "credits,keywords", "language": "en-US"},
        )


def load_cache(cache_path: Path) -> dict[int, dict[str, Any]]:
    """Wczytuje wcześniej pobrane rekordy TMDB z pliku JSONL (klucz: movieId)."""
    cache: dict[int, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[int(rec["movieId"])] = rec
            except Exception:
                continue
    return cache


def extract_fields_from_tmdb(
    row: pd.Series, tmdb_id: int | None, details: dict[str, Any] | None
) -> dict[str, Any]:
    """Przetwarza surową odpowiedź TMDB na ustrukturyzowany rekord filmu."""
    ml_genres = list(row["genres_list"]) if row["genres_list"] is not None else []

    if not details:
        overview = ""
        keywords: list[str] = []
        tmdb_genres: list[str] = []
        cast: list[str] = []
        director = ""
        poster_path = ""
    else:
        overview = str(details.get("overview") or "").strip()
        tmdb_genres = [
            g["name"].strip() for g in details.get("genres", []) if g.get("name")
        ]
        kw_container = details.get("keywords") or {}
        kw_list = kw_container.get("keywords") or kw_container.get("results") or []
        keywords = [k["name"].strip() for k in kw_list if k.get("name")][:15]

        credits = details.get("credits") or {}
        cast_raw = credits.get("cast") or []
        cast = [c["name"].strip() for c in cast_raw if c.get("name")][:5]

        crew_raw = credits.get("crew") or []
        directors = [
            c["name"].strip()
            for c in crew_raw
            if c.get("job") == "Director" and c.get("name")
        ]
        director = ", ".join(dict.fromkeys(directors))
        poster_path = str(details.get("poster_path") or "").strip()

    # Połączenie gatunków z MovieLens i TMDB (z zachowaniem kolejności bez duplikatów)
    combined_genres = list(dict.fromkeys(ml_genres + tmdb_genres))

    # Złożenie tekstu do embeddingu Transformer (w języku angielskim, spójnie z modelem)
    parts = [f"{row['title']}."]
    if combined_genres:
        parts.append(f"Genres: {', '.join(combined_genres)}.")
    if director:
        parts.append(f"Director: {director}.")
    if cast:
        parts.append(f"Cast: {', '.join(cast)}.")
    if keywords:
        parts.append(f"Keywords: {', '.join(keywords)}.")
    if overview:
        parts.append(f"Overview: {overview}")

    embedding_text = " ".join(parts)

    return {
        "movieId": int(row["movieId"]),
        "tmdbId": int(tmdb_id) if tmdb_id is not None and not pd.isna(tmdb_id) else None,
        "title": str(row["title"]),
        "clean_title": str(row["clean_title"]),
        "year": int(row["year"]) if pd.notna(row["year"]) else None,
        "genres": combined_genres,
        "overview": overview,
        "keywords": keywords,
        "cast": cast,
        "director": director,
        "poster_path": poster_path,
        "embedding_text": embedding_text,
        "has_full_overview": len(overview) >= 20,
    }


def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key or api_key == "twoj_klucz_api_tutaj":
        print(
            "[UWAGA] Brak klucza TMDB_API_KEY w pliku .env!\n"
            "Aby pobrać opisy filmów, słowa kluczowe i obsadę z TMDB:\n"
            "  1. Skopiuj plik .env.example do .env\n"
            "  2. Wklej swój klucz API z https://www.themoviedb.org/settings/api\n"
            "  3. Uruchom ponownie: .\\.venv\\Scripts\\python.exe src\\enrich_tmdb.py"
        )
        return

    movies_base_path = PROCESSED_DIR / "movies_base.parquet"
    if not movies_base_path.exists():
        raise FileNotFoundError(
            f"Nie znaleziono pliku {movies_base_path}. Najpierw uruchom src/download_movielens.py."
        )

    movies_df = pd.read_parquet(movies_base_path)
    cache = load_cache(CACHE_PATH)
    print(f"[INFO] Wczytano {len(movies_df):,} filmów. W pamięci podręcznej TMDB: {len(cache):,} filmów.")

    client = TMDBClient(api_key)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    todo_rows = [
        row for _, row in movies_df.iterrows() if int(row["movieId"]) not in cache
    ]

    if todo_rows:
        print(f"[INFO] Pobieranie danych z TMDB dla {len(todo_rows):,} filmów (16 wątków) ...")

        def process_one(row: pd.Series) -> dict[str, Any]:
            tmdb_id = int(row["tmdbId"]) if pd.notna(row["tmdbId"]) else None
            details = None
            if tmdb_id is not None:
                details = client.fetch_movie_details(tmdb_id)
            # Jeśli brak tmdbId lub film pod starym tmdbId został usunięty w TMDB -> szukamy po tytule i roku
            if details is None:
                found_id = client.search_movie(str(row["clean_title"]), row["year"])
                if found_id is not None:
                    tmdb_id = found_id
                    details = client.fetch_movie_details(tmdb_id)
            return extract_fields_from_tmdb(row, tmdb_id, details)

        with CACHE_PATH.open("a", encoding="utf-8") as cache_file:
            with ThreadPoolExecutor(max_workers=16) as executor:
                futures = {executor.submit(process_one, row): row for row in todo_rows}
                for future in tqdm(as_completed(futures), total=len(futures), desc="TMDB API"):
                    rec = future.result()
                    cache[rec["movieId"]] = rec
                    cache_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    cache_file.flush()

    # Budowa pełnej tabeli w kolejności movieId z MovieLens 1M
    enriched_records = [cache[int(m_id)] for m_id in movies_df["movieId"] if int(m_id) in cache]
    enriched_df = pd.DataFrame(enriched_records).sort_values("movieId").reset_index(drop=True)

    out_path = PROCESSED_DIR / "movies_enriched.parquet"
    enriched_df.to_parquet(out_path, index=False)

    n_total = len(enriched_df)
    n_matched = enriched_df["tmdbId"].notna().sum()
    n_overview = enriched_df["has_full_overview"].sum()
    n_keywords = enriched_df["keywords"].apply(lambda x: len(x) > 0).sum()
    n_director = enriched_df["director"].apply(lambda x: bool(str(x).strip())).sum()

    print("\n" + "=" * 60)
    print("PODSUMOWANIE WZBOGACENIA DANYCH Z TMDB")
    print("=" * 60)
    print(f"Zapisano plik:                 {out_path}")
    print(f"Liczba filmów ogółem:          {n_total:,}")
    print(f"Dopasowane w TMDB (tmdbId):    {n_matched:,} ({n_matched / n_total * 100:.1f}%)")
    print(f"Pełny opis fabuły (overview):  {n_overview:,} ({n_overview / n_total * 100:.1f}%)")
    print(f"Słowa kluczowe (keywords):     {n_keywords:,} ({n_keywords / n_total * 100:.1f}%)")
    print(f"Informacja o reżyserze:        {n_director:,} ({n_director / n_total * 100:.1f}%)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
