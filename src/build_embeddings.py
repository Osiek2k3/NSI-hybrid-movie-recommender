"""
Generowanie embeddingów Transformer (all-MiniLM-L6-v2, 384 wymiary)
dla wszystkich filmów z pliku data/processed/movies_enriched.parquet.

Dla każdego filmu składany jest tekst z tytułu, gatunków, reżysera, obsady,
słów kluczowych i opisu fabuły z TMDB (pole embedding_text).
Wektory są normalizowane (L2 = 1), dzięki czemu iloczyn skalarny odpowiada
podobieństwu kosinusowemu.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    movies_path = PROCESSED_DIR / "movies_enriched.parquet"
    if not movies_path.exists():
        raise FileNotFoundError(
            f"Brak pliku {movies_path}. Uruchom najpierw src/enrich_tmdb.py."
        )

    movies = pd.read_parquet(movies_path).sort_values("movieId").reset_index(drop=True)
    texts = movies["embedding_text"].fillna(movies["title"]).tolist()
    movie_ids = movies["movieId"].to_numpy(dtype=np.int32)

    print(f"[INFO] Ładowanie modelu Transformer: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"[INFO] Generowanie embeddingów dla {len(texts):,} filmów ...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    emb_path = PROCESSED_DIR / "movie_embeddings.npy"
    ids_path = PROCESSED_DIR / "movie_ids.npy"
    np.save(emb_path, embeddings)
    np.save(ids_path, movie_ids)

    print("\n" + "=" * 60)
    print("PODSUMOWANIE EMBEDDINGÓW TRANSFORMER")
    print("=" * 60)
    print(f"Model:                  {MODEL_NAME}")
    print(f"Kształt macierzy:       {embeddings.shape} (filmy × wymiary)")
    print(f"Zapisano wektory:       {emb_path}")
    print(f"Zapisano ID filmów:     {ids_path}")

    # Szybki test semantyczny: filmy najbardziej podobne do Toy Story (movieId=1) i The Matrix (movieId=2571)
    id_to_idx = {int(m_id): idx for idx, m_id in enumerate(movie_ids)}
    for sample_id in (1, 2571):
        if sample_id in id_to_idx:
            idx = id_to_idx[sample_id]
            sims = embeddings @ embeddings[idx]
            top_indices = np.argsort(-sims)[1:6]
            sample_title = movies.loc[idx, "title"]
            similar_titles = [
                f"{movies.loc[i, 'title']} ({sims[i]:.3f})" for i in top_indices
            ]
            print(f"\nNajbardziej podobne treściowo do '{sample_title}':")
            for t in similar_titles:
                print(f"  - {t}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
