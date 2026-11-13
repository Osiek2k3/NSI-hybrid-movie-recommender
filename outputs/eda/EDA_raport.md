# Eksploracyjna Analiza Danych (EDA): MovieLens 1M + TMDB

Dokument zawiera kompletną analizę wszystkich tabel, cech (atrybutów), rozkładów statystycznych oraz zjawiska długiego ogona (*long tail*) dla zbioru **MovieLens 1M** wzbogaconego o metadane z **The Movie Database (TMDB API)**.

---

## 1. Zestawienie wszystkich tabel i cech (słownik danych)

### 1.1. Tabela ocen: `ratings.dat` (`data/processed/ratings.parquet`)
- **Liczba rekordów:** `1 000 209`
- **Braki danych (NULL/NaN):** `0` (100% kompletności)

| Nazwa cechy | Typ danych | Zakres / Wartości | Opis i rola w systemie rekomendacyjnym |
| :--- | :--- | :--- | :--- |
| `userId` | `int64` | `1 – 6 040` (`6 040` unikalnych) | Identyfikator użytkownika oceniającego film. |
| `movieId` | `int64` | `1 – 3 952` (`3 706` unikalnych) | Identyfikator ocenionego filmu (uwaga: `177` filmów z `movies.dat` nie ma żadnej oceny). |
| `rating` | `int64` | `{1, 2, 3, 4, 5}` | Ocena w skali 1–5 gwiazdek (jawna informacja zwrotna; oceny $\ge 4$ traktujemy jako trafne/pozytywne). |
| `timestamp` | `int64` | `956703932 – 1046454590` (`25.04.2000 – 28.02.2003`) | Czas wystawienia oceny w sekundach od `1970-01-01 UTC` – służy do chronologicznego podziału trening/test oraz symulacji onboardingu. |

---

### 1.2. Tabela użytkowników: `users.dat` (`data/processed/users.parquet`)
- **Liczba rekordów:** `6 040`
- **Braki danych:** `0`

| Nazwa cechy | Typ danych | Zakres / Wartości | Opis i rola w systemie rekomendacyjnym |
| :--- | :--- | :--- | :--- |
| `userId` | `int64` | `1 – 6 040` | Klucz główny użytkownika. |
| `gender` | `string` | `'M'` (`4 331`, `71,7%`), `'F'` (`1 709`, `28,3%`) | Płeć użytkownika. |
| `age` | `int64` | `{1, 18, 25, 35, 45, 50, 56}` (7 przedziałów) | Kod przedziału wiekowego (`1` = poniżej 18 lat, `25` = 25–34 lata, `56` = 56+ lat). |
| `occupation` | `int64` | `0 – 20` (21 kategorii zawodowych) | Kod zawodu wg słownika GroupLens (np. `4` = student, `7` = menedżer, `12` = programista). |
| `zipCode` | `string` | `3 439` unikalnych kodów | Kod pocztowy w USA (dane zanonimizowane). |

---

### 1.3. Tabela filmów połączona z TMDB: `movies_enriched.parquet`
- **Liczba rekordów:** `3 883` filmów w katalogu MovieLens 1M
- **Źródło podstawowe:** `movies.dat` (`movieId`, `title`, `genres`) + `links.csv` (`ml-latest-small`) + zapytania `GET /3/movie/{tmdbId}?append_to_response=credits,keywords`

| Nazwa cechy | Typ danych | Pokrycie (kompletność) | Statystyki / Przykłady | Rola w projekcie |
| :--- | :--- | :---: | :--- | :--- |
| `movieId` | `int64` | `3 883 / 3 883` (`100,0%`) | `1 – 3 952` (luki w numeracji oryginalnego zbioru) | Klucz główny filmu w MovieLens. |
| `tmdbId` | `Int64` | `3 872 / 3 883` (`99,72%`) | `2 941` z `links.csv` + `931` wyszukanych po tytule i roku | Klucz filmu w bazie TMDB. |
| `title` | `string` | `3 883 / 3 883` (`100,0%`) | np. `Toy Story (1995)` | Oryginalny tytuł z rokiem wydania. |
| `clean_title` | `string` | `3 883 / 3 883` (`100,0%`) | np. `The Matrix` (zamiast `Matrix, The (1999)`) | Oczyszczony tytuł do wyszukiwania i UI. |
| `year` | `Int64` | `3 883 / 3 883` (`100,0%`) | `1919 – 2000` (mediana roku produkcji: `1994`) | Rok premiery filmu wyodrębniony z tytułu. |
| `genres` | `list[str]` | `3 883 / 3 883` (`100,0%`) | Średnio `2,52` gatunku na film (18 gatunków ML + TMDB) | Embedding treściowy, dywersyfikacja Cold Start i wyjaśnienia. |
| `overview` | `string` | `3 869 / 3 883` (`99,64%`) | Śr. `47,7` słów (`273` znaki), mediana `44` słowa | Główny opis fabuły (ang.) do modelu Transformer `all-MiniLM-L6-v2`. |
| `keywords` | `list[str]` | `3 583 / 3 883` (`92,27%`) | Śr. `8,77` słów kluczowych na film (mediana `9`, maks. `15`) | Doprecyzowanie motywów fabularnych w embeddingu i wyjaśnieniach LLM. |
| `cast` | `list[str]` | `3 837 / 3 883` (`98,82%`) | Top-5 aktorów (najczęściej: Robert De Niro – 30 filmów) | Fakty do wyjaśnień („występuje tu także...”) oraz embeddingu. |
| `director` | `string` | `3 867 / 3 883` (`99,59%`) | Najczęściej: Alfred Hitchcock (55), Woody Allen (20) | Fakty do wyjaśnień („ten sam reżyser co...”) oraz embeddingu. |
| `poster_path` | `string` | `3 862 / 3 883` (`99,46%`) | Ścieżka względna TMDB (`/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg`) | Wyświetlanie plakatów w prototypie aplikacji. |
| `embedding_text` | `string` | `3 883 / 3 883` (`100,0%`) | Średnio `86,0` słów na film | Pełny tekst kodowany przez `all-MiniLM-L6-v2` do wektora $\mathbb{R}^{384}$. |

---

## 2. Statystyki ocen (`ratings.dat`)

- **Średnia ocena globalna:** `3,5816` (odchylenie standardowe: `1,1171`)
- **Mediana:** `4,0` gwiazdki; **Dominanta (moda):** `4` gwiazdki (`34,89%` wszystkich ocen)
- **Odsetek ocen pozytywnych ($\ge 4$ gwiazdki):** `57,52%` (`575 281` ocen)

| Ocena (gwiazdki) | Liczba ocen | Udział (%) | Interpretacja w projekcie |
| :---: | ---: | ---: | :--- |
| **1** | 56 174 | 5,62% | Silne odrzucenie |
| **2** | 107 557 | 10,75% | Ocena negatywna |
| **3** | 261 197 | 26,11% | Ocena neutralna |
| **4** | 348 971 | 34,89% | Pozytywne dopasowanie (*relevant*) |
| **5** | 226 310 | 22,63% | Ulubione filmy (*highly relevant*) |

---

## 3. Rzadkość macierzy (Sparsity) i zjawisko „długiego ogona” (Long Tail)

- **Wymiar macierzy użytkownik × film:** `6 040 × 3 883 = 23 453 320` możliwych par.
- **Wypełnienie macierzy (gęstość):** `4,26%` (dla `3 706` ocenionych filmów: `4,47%`), co oznacza **rzadkość (sparsity) na poziomie `95,74%`**.
- **Filmy bez żadnej oceny (Cold Item):** W pliku `movies.dat` znajduje się **`177` filmów (`4,56%` katalogu)**, które nie mają ani jednej oceny w `ratings.dat`. Dla czystego filtrowania kolaboratywnego (ALS/SVD) są one całkowicie niewidoczne, natomiast metoda oparta na treści (embeddingi Transformer z opisów TMDB) potrafi je normalnie reprezentować i polecać.
- **Koncentracja popularności (zasada Pareto):**
  - Top **10% najpopularniejszych filmów** (`388` tytułów) skupia aż **45,7%** wszystkich ocen.
  - Top **20% najpopularniejszych filmów** (`776` tytułów) skupia **66,6%** wszystkich ocen.
  - Zaledwie **11,6% filmów** wystarczy do pokrycia połowy (`50%`) wszystkich wystawionych ocen, a **30,4% filmów** generuje `80%` całego ruchu.

### Top-10 najczęściej ocenianych filmów w MovieLens 1M:
| Miejsce | `movieId` | Tytuł filmu | Reżyser (z TMDB) | Liczba ocen | Średnia ocena | Odchylenie std |
| :---: | :---: | :--- | :--- | ---: | :---: | :---: |
| 1 | 2858 | *American Beauty (1999)* | Sam Mendes | 3 428 | 4,32 | 0,98 |
| 2 | 260 | *Star Wars: Episode IV - A New Hope (1977)* | George Lucas | 2 991 | 4,45 | 0,80 |
| 3 | 1196 | *Star Wars: Episode V - The Empire Strikes Back (1980)* | Irvin Kershner | 2 990 | 4,29 | 0,84 |
| 4 | 1210 | *Star Wars: Episode VI - Return of the Jedi (1983)* | Richard Marquand | 2 883 | 4,02 | 0,93 |
| 5 | 480 | *Jurassic Park (1993)* | Steven Spielberg | 2 672 | 3,76 | 0,97 |
| 6 | 2028 | *Saving Private Ryan (1998)* | Steven Spielberg | 2 653 | 4,34 | 0,85 |
| 7 | 589 | *Terminator 2: Judgment Day (1991)* | James Cameron | 2 649 | 4,06 | 0,90 |
| 8 | 2571 | *Matrix, The (1999)* | Lana & Lilly Wachowski | 2 590 | 4,32 | 0,93 |
| 9 | 1270 | *Back to the Future (1985)* | Robert Zemeckis | 2 583 | 3,99 | 0,87 |
| 10 | 593 | *Silence of the Lambs, The (1991)* | Jonathan Demme | 2 578 | 4,35 | 0,83 |

---

## 4. Charakterystyka użytkowników (aktywność i demografia)

### 4.1. Rozkład liczby ocen na użytkownika
- **Minimum:** `20` ocen (próg wejścia do zbioru MovieLens 1M)
- **Kwartyl dolny ($Q_1$, 25%):** `44` oceny
- **Mediana ($Q_2$, 50%):** `96` ocen
- **Średnia:** `165,6` ocen (odchylenie standardowe: `192,7`)
- **Kwartyl górny ($Q_3$, 75%):** `208` ocen
- **90. percentyl ($P_{90}$):** `400` ocen
- **Maksimum:** `2 314` ocen (użytkownik `userId = 4169`)

### 4.2. Rozkład płci i wieku
- **Mężczyźni (`M`):** `4 331` użytkowników (`71,7%`), oddali `753 769` ocen (`75,4%`), średnia ocena: `3,569`.
- **Kobiety (`F`):** `1 709` użytkowniczek (`28,3%`), oddały `246 440` ocen (`24,6%`), średnia ocena: `3,620`.

| Kod wieku | Przedział wieku | Liczba użytkowników | Udział (%) | Liczba ocen | Średnia ocena |
| :---: | :--- | ---: | ---: | ---: | :---: |
| `1` | Poniżej 18 lat (`<18`) | 222 | 3,68% | 27 211 | 3,550 |
| `18` | `18–24` lata | 1 103 | 18,26% | 183 536 | 3,508 |
| `25` | `25–34` lata | 2 096 | 34,70% | 395 556 | 3,545 |
| `35` | `35–44` lata | 1 193 | 19,75% | 199 003 | 3,618 |
| `45` | `45–49` lat | 550 | 9,11% | 83 633 | 3,638 |
| `50` | `50–55` lat | 496 | 8,21% | 72 490 | 3,715 |
| `56` | `56+` lat | 380 | 6,29% | 38 780 | 3,767 |

*(Widać wyraźną prawidłowość: wraz z wiekiem użytkowników rośnie średnia wystawiana ocena – od `3,51` w grupie 18–24 lata do `3,77` w grupie 56+).*

---

## 5. Rozkład 18 gatunków filmowych w MovieLens 1M

| Gatunek | Liczba filmów | Odsetek katalogu (%) | Liczba ocen | Średnia ocena | Odchylenie std ocen |
| :--- | ---: | ---: | ---: | :---: | :---: |
| **Drama** | 1 603 | 41,28% | 354 529 | 3,766 | 1,046 |
| **Comedy** | 1 200 | 30,90% | 356 580 | 3,522 | 1,121 |
| **Action** | 503 | 12,95% | 257 457 | 3,491 | 1,133 |
| **Thriller** | 492 | 12,67% | 189 680 | 3,570 | 1,107 |
| **Romance** | 471 | 12,13% | 147 523 | 3,607 | 1,067 |
| **Horror** | 343 | 8,83% | 76 386 | 3,215 | 1,226 |
| **Adventure** | 283 | 7,29% | 133 953 | 3,477 | 1,129 |
| **Sci-Fi** | 276 | 7,11% | 157 294 | 3,467 | 1,157 |
| **Children's** | 251 | 6,46% | 72 186 | 3,422 | 1,161 |
| **Crime** | 211 | 5,43% | 79 541 | 3,709 | 1,078 |
| **War** | 143 | 3,68% | 68 527 | 3,893 | 1,065 |
| **Documentary** | 127 | 3,27% | 7 910 | 3,933 | 1,033 |
| **Musical** | 114 | 2,94% | 41 533 | 3,666 | 1,101 |
| **Mystery** | 106 | 2,73% | 40 178 | 3,668 | 1,087 |
| **Animation** | 105 | 2,70% | 43 293 | 3,685 | 1,082 |
| **Fantasy** | 68 | 1,75% | 36 301 | 3,447 | 1,133 |
| **Western** | 68 | 1,75% | 20 683 | 3,638 | 1,100 |
| **Film-Noir** | 44 | 1,13% | 18 261 | 4,075 | 0,933 |

*(Najbardziej polaryzującym gatunkiem o najwyższym odchyleniu standardowym i najniższej średniej jest **Horror** (`3,215 ± 1,226`), natomiast najwyższe oceny zbierają klasyki **Film-Noir** (`4,075`), **Documentary** (`3,933`) oraz **War** (`3,893`)).*
