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

---

## 6. Analiza braków danych (`NULL`), anomalii oraz koniecznych ingerencji i symulacji

### 6.1. Czy w danych występują wartości `NULL` / braki i anomalie?

1. **Surowe tabele `ratings.dat` i `users.dat`:**
   - W samych plikach tekstowych **nie ma formalnych wartości `NULL` (`0` braków)** – każdy z `1 000 209` wierszy ocen i `6 040` wierszy użytkowników ma komplet kolumn.
   - Jednak na poziomie **macierzy interakcji użytkownik × film** (`6 040 × 3 883 = 23 453 320` par) aż **`22 453 111` komórek (`95,74%`) jest pustych (`NULL`)** – przeciętny użytkownik ocenił tylko `4,26%` katalogu.
2. **Anomalie i luki w `movies.dat` (MovieLens 1M):**
   - **Luki w kluczach `movieId`:** Identyfikatory filmów sięgają od `1` do `3 952`, ale w pliku `movies.dat` znajdują się tylko `3 883` filmy – **`69` numerów ID w ogóle nie istnieje** (zostały usunięte przez twórców zbioru przed publikacją).
   - **Filmy bez ani jednej oceny (*Cold Items*):** Spośród `3 883` filmów w katalogu `movies.dat` aż **`177` filmów (`4,56%`) nie pojawia się ani razu w `ratings.dat`** (np. *Guardian Angel (1994)*, *Gospa (1995)*, *New York Cop (1996)*). Dla czystego filtrowania kolaboratywnego (ALS/SVD) te filmy mają wektor zerowy i nigdy nie zostaną polecone.
   - **Niespójny zapis tytułów i roczników:** Tytuły mają przestawione rodzajniki na koniec (np. `Matrix, The (1999)`), często zawierają drugi tytuł obcojęzyczny w nawiasie (np. `Silence of the Palace, The (Saimt el Qusur) (1994)`), a w `movieId = 1001` (`Associate, The (L'Associe)(1982)`) brakuje spacji przed rokiem wydania.
3. **Braki przy łączeniu z `links.csv` (`ml-latest-small`) oraz z API TMDB:**
   - W `links.csv` z `ml-latest-small` brakowało powiązania `tmdbId` dla **`937` filmów** z MovieLens 1M (`24,1%`), a dla kolejnych **`5` filmów** przypisany `tmdbId` wskazywał na inny film (rozbieżność tytułu/roku po weryfikacji) – łącznie **`942` braki (`NULL`)** wymagały wyszukania w API TMDB po oczyszczonym tytule i roku.
   - Po automatycznym odpytaniu wyszukiwarki TMDB (`GET /3/search/movie`) pozostało jedynie:
     - **`11` filmów (`0,28%`) bez `tmdbId`** (niszowe produkcje i dokumenty z lat 1949–1997, np. *Aiqing wansui (1994)*, *Daughter of Dr. Jeckyll (1957)*),
     - **`14` filmów (`0,36%`) z pustym opisem fabuły (`overview = ""`)** (w tym `11` niedopasowanych + `3` filmy istniejące w TMDB bez angielskiego opisu: *Criminals (1996)*, *Diebinnen (1995)*, *Mondo (1996)*),
     - **`300` filmów (`7,73%`) bez słów kluczowych (`keywords = []`)**,
     - **`26` filmów (`0,67%`) bez obsady (`cast = []`)**,
     - **`16` filmów (`0,41%`) bez przypisanego reżysera (`director = ""`)**,
     - **`21` filmów (`0,54%`) bez plakatu (`poster_path = ""`)**.

---

### 6.2. Dlaczego i jak musimy ingerować w dane oraz symulować wybrane scenariusze?

| Obszar problemu w surowych danych | Dlaczego musimy zaingerować / symulować? | Jak dokładnie rozwiązujemy to w potoku danych? |
| :--- | :--- | :--- |
| **1. Brak cech treściowych w MovieLens 1M** | W `movies.dat` są wyłącznie tytuł, rok i ogólne gatunki (brak fabuły, słów kluczowych, reżysera i aktorów). Z samych gatunków nie da się zbudować precyzyjnych embeddingów semantycznych ani wyjaśnień LLM. | Wzbogacamy zbiór zewnętrznym API **TMDB**, łącząc `links.csv` z wyszukiwaniem po oczyszczonym tytule i roku (`99,72%` pokrycia). |
| **2. Imputacja braków tekstowych z TMDB (`overview`, `keywords`)** | Wstawienie wartości `NULL` lub napisu `"None"` do tekstu kodowanego przez sieć Transformer (`all-MiniLM-L6-v2`) zniekształciłoby wektor embeddingu. | Stosujemy **imputację strukturalną (warunkowe składanie tekstu)**: jeśli film nie ma `overview` (14 filmów) lub `keywords` (300 filmów), `embedding_text` jest dynamicznie budowany wyłącznie z istniejących pól (tytułu, roku, gatunków MovieLens i obsady/reżysera). |
| **3. Brak nowych i mało aktywnych użytkowników w MovieLens 1M (próg $\ge 20$ ocen)** | Twórcy MovieLens 1M celowo odfiltrowali wszystkich użytkowników mających mniej niż 20 ocen (`min = 20`). Bez ingerencji w dane **nie istniałby ani jeden użytkownik** w grupach Cold Start ($N=0$), $1\text{–}5$ ocen oraz $6\text{–}20$ ocen! | **Symulacja kontrolowana (ukrywanie historii):**<br>1) **Eksperyment Cold Start:** losujemy 15% użytkowników (`906` osób), całkowicie usuwamy ich z treningu CF i odsłaniamy chronologicznie tylko $N \in \{0, 1, 3, 5, 10\}$ pierwszych filmów ocenionych na 4–5★ jako wybór z onboardingu.<br>2) **Grupy aktywności (`1–5` i `6–20` ocen):** dla wylosowanych podgrup użytkowników przycinamy historię treningową do pierwszych $1..5$ lub $6..20$ chronologicznych ocen. |
| **4. Przeciek informacji z przyszłości (*Data Leakage*) przy losowym podziale** | Zwykły losowy podział `train_test_split` sprawiłby, że model uczyłby się na ocenach wystawionych w 2002 roku, aby przewidywać filmy oglądane przez tego samego użytkownika w 2000 roku. | Stosujemy **podział chronologiczny po polu `timestamp`** wewnątrz każdego użytkownika (starsze 80% ocen do nauki, najnowsze 20% ocen do testu). |
| **5. Przesunięcie skali ocen i brak jawnych negatywów w rankingu** | Średnia ocena w zbiorze to aż `3,58`, a ocena `1★` lub `2★` oznacza, że film się **nie podobał**. Wrzucenie wszystkich ocen do ALS traktowałoby obejrzany znienawidzony film jako sukces rekomendacji. | Przeprowadzamy **filtrację i ważenie sygnału**: za trafne rekomendacje (*relevant*) w ewaluacji oraz w macierzy implicit ALS uznajemy wyłącznie oceny pozytywne ($r_{ui} \ge 4$, stanowiące `57,52%` zbioru), z wagą pewności rosnącą wraz z oceną. |

