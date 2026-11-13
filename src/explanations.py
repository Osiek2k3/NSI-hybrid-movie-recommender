"""
Moduł generowania wyjaśnień rekomendacji po polsku (oparty na faktach z TMDB oraz wagach modelu hybrydowego).

Zgodnie z założeniami projektu wykorzystuje:
1. Wagę α(n) z hybrydy – informuje, w jakim stopniu rekomendacja wynika z podobieństwa treści
   (embeddingi Transformer), a w jakim z zachowań podobnych użytkowników (ALS).
2. Przewidywaną ocenę gwiazdkową (1–5) z modelu SVD.
3. Konkretne fakty z TMDB łączące polecany film z ulubionymi filmami użytkownika:
   - wspólny reżyser ("ten sam reżyser co..."),
   - wspólna obsada aktorska,
   - wspólne słowa kluczowe (keywords) i gatunki,
   - najbardziej zbliżone semantycznie filmy z historii użytkownika (wg embeddingów Transformer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ExplanationContext:
    user_id: int
    n_user_ratings: int
    recommended_movie_id: int
    recommended_title: str
    alpha_cf: float
    content_weight: float
    predicted_svd_rating: float
    top_similar_watched: list[tuple[str, float]]
    shared_directors: list[tuple[str, str]]  # (director, watched_title)
    shared_actors: list[tuple[str, str]]     # (actor, watched_title)
    shared_keywords: list[str]
    shared_genres: list[str]
    prompt_for_llm: str
    explanation_pl: str


class RecommendationExplainer:
    """Buduje ustrukturyzowane fakty, prompt dla modelu LLM oraz wyjaśnienie w języku polskim."""

    def __init__(
        self,
        movies_df: pd.DataFrame,
        embeddings: np.ndarray,
        movie_ids: np.ndarray,
    ) -> None:
        self.movies_df = movies_df.set_index("movieId", drop=False)
        self.embeddings = embeddings
        self.movie_ids = movie_ids
        self.id_to_idx = {int(mid): idx for idx, mid in enumerate(movie_ids)}

    def _to_list(self, val: Any) -> list[str]:
        if val is None:
            return []
        if isinstance(val, (list, np.ndarray)):
            return [str(x).strip() for x in val if str(x).strip()]
        s = str(val).strip()
        return [x.strip() for x in s.split(",") if x.strip()] if s else []

    def explain(
        self,
        user_id: int,
        liked_movie_ids: list[int],
        recommended_movie_id: int,
        alpha_cf: float,
        predicted_svd_rating: float,
    ) -> ExplanationContext:
        rec_row = self.movies_df.loc[int(recommended_movie_id)]
        rec_title = str(rec_row["title"])
        rec_director_set = {
            d.strip() for d in str(rec_row.get("director") or "").split(",") if d.strip()
        }
        rec_cast = set(self._to_list(rec_row.get("cast")))
        rec_keywords = set(self._to_list(rec_row.get("keywords")))
        rec_genres = set(self._to_list(rec_row.get("genres")))

        rec_idx = self.id_to_idx[int(recommended_movie_id)]
        rec_vec = self.embeddings[rec_idx]

        # Szukamy najbardziej podobnych filmów z historii użytkownika oraz wspólnych faktów TMDB
        sim_watched: list[tuple[str, float]] = []
        shared_directors: list[tuple[str, str]] = []
        shared_actors: list[tuple[str, str]] = []
        matched_keywords: set[str] = set()
        matched_genres: set[str] = set()

        for wid in liked_movie_ids:
            if int(wid) not in self.movies_df.index or int(wid) not in self.id_to_idx:
                continue
            w_row = self.movies_df.loc[int(wid)]
            w_title = str(w_row["title"])
            w_idx = self.id_to_idx[int(wid)]
            sim = float(np.dot(rec_vec, self.embeddings[w_idx]))
            sim_watched.append((w_title, sim))

            w_dirs = {
                d.strip() for d in str(w_row.get("director") or "").split(",") if d.strip()
            }
            for d in rec_director_set.intersection(w_dirs):
                shared_directors.append((d, w_title))

            w_cast = set(self._to_list(w_row.get("cast")))
            for a in rec_cast.intersection(w_cast):
                shared_actors.append((a, w_title))

            w_kw = set(self._to_list(w_row.get("keywords")))
            matched_keywords.update(rec_keywords.intersection(w_kw))

            w_gen = set(self._to_list(w_row.get("genres")))
            matched_genres.update(rec_genres.intersection(w_gen))

        sim_watched.sort(key=lambda x: x[1], reverse=True)
        top_similar = sim_watched[:3]

        content_weight = 1.0 - alpha_cf
        n_ratings = len(liked_movie_ids)

        # Budowa promptu dla LLM (z uziemieniem w faktach, bez halucynacji)
        prompt_lines = [
            "Jesteś asystentem hybrydowego systemu rekomendacji filmów. Napisz zwięzłe (2-3 zdania), "
            "naturalne wyjaśnienie po polsku, dlaczego polecamy ten film użytkownikowi, "
            "korzystając WYŁĄCZNIE z poniższych faktów:",
            f"- Polecany film: {rec_title}",
            f"- Liczba wysoko ocenionych filmów użytkownika: {n_ratings}",
            f"- Udział metod w hybrydzie: {content_weight * 100:.0f}% dopasowanie treści (embeddingi Transformer), "
            f"{alpha_cf * 100:.0f}% filtrowanie kolaboratywne (ALS – zachowania podobnych widzów)",
            f"- Przewidywana ocena z modelu SVD: {predicted_svd_rating:.2f}/5.00 gwiazdek",
            f"- Najbardziej zbliżone tematycznie filmy z historii użytkownika: "
            + ", ".join(f"{t} (podobieństwo {s:.2f})" for t, s in top_similar),
        ]
        if shared_directors:
            d_name, w_t = shared_directors[0]
            prompt_lines.append(f"- Wspólny reżyser: {d_name} (wyreżyserował też lubiany przez użytkownika film {w_t})")
        if shared_actors:
            a_name, w_t = shared_actors[0]
            prompt_lines.append(f"- Wspólny aktor: {a_name} (grał też w {w_t})")
        if matched_keywords:
            prompt_lines.append(f"- Wspólne motywy (TMDB keywords): {', '.join( sorted(matched_keywords)[:5] )}")
        if matched_genres:
            prompt_lines.append(f"- Wspólne gatunki: {', '.join( sorted(matched_genres)[:4] )}")

        prompt_for_llm = "\n".join(prompt_lines)

        # Wygenerowanie wyjaśnienia po polsku na podstawie faktów
        if alpha_cf < 0.35:
            mechanism_sentence = (
                f"Ponieważ Twój profil opiera się na krótkiej historii ({n_ratings} ocen), "
                f"rekomendacja bazuje głównie na analizie treści ({content_weight * 100:.0f}% udziału embeddingów Transformer, "
                f"{alpha_cf * 100:.0f}% udziału ALS)."
            )
        elif alpha_cf <= 0.65:
            mechanism_sentence = (
                f"Rekomendacja w zrównoważony sposób łączy analizę treści ({content_weight * 100:.0f}%) "
                f"oraz wybory widzów o podobnym guście ({alpha_cf * 100:.0f}% udziału ALS)."
            )
        else:
            mechanism_sentence = (
                f"Dzięki bogatej historii ocen ({n_ratings} pozytywnych filmów) system oparł się przede wszystkim "
                f"na filtrowaniu kolaboratywnym ALS ({alpha_cf * 100:.0f}% wagi), uzupełnionym o podobieństwo treści ({content_weight * 100:.0f}%)."
            )

        fact_parts: list[str] = []
        if top_similar:
            top_titles = " oraz ".join(f"„{t}”" for t, _ in top_similar[:2])
            fact_parts.append(f"Film „{rec_title}” jest silnie powiązany tematycznie z wysoko ocenionymi przez Ciebie tytułami {top_titles}")
        if shared_directors:
            d_name, w_t = shared_directors[0]
            fact_parts.append(f"za kamerą stanął ten sam reżyser ({d_name}, twórca „{w_t}”)")
        if shared_actors:
            a_name, w_t = shared_actors[0]
            fact_parts.append(f"w obsadzie ponownie pojawia się {a_name} (znany z „{w_t}”)")
        if matched_keywords:
            kws = ", ".join(sorted(matched_keywords)[:3])
            fact_parts.append(f"łączy je wspólna tematyka ({kws})")
        elif matched_genres:
            gens = ", ".join(sorted(matched_genres)[:3])
            fact_parts.append(f"reprezentuje Twoje ulubione gatunki ({gens})")

        facts_sentence = (
            ", a ponadto ".join(fact_parts[:2]) + "."
            if len(fact_parts) >= 2
            else (fact_parts[0] + "." if fact_parts else "")
        )
        svd_sentence = f"Przewidywana ocena dla Ciebie według modelu SVD wynosi {predicted_svd_rating:.1f}/5,0 gwiazdek."

        explanation_pl = " ".join(
            part for part in [mechanism_sentence, facts_sentence, svd_sentence] if part
        )

        return ExplanationContext(
            user_id=user_id,
            n_user_ratings=n_ratings,
            recommended_movie_id=recommended_movie_id,
            recommended_title=rec_title,
            alpha_cf=alpha_cf,
            content_weight=content_weight,
            predicted_svd_rating=predicted_svd_rating,
            top_similar_watched=top_similar,
            shared_directors=shared_directors,
            shared_actors=shared_actors,
            shared_keywords=sorted(matched_keywords)[:6],
            shared_genres=sorted(matched_genres)[:5],
            prompt_for_llm=prompt_for_llm,
            explanation_pl=explanation_pl,
        )
