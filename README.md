# Hotel Analytics ML

Projekt realizowany w ramach kursu eksploracji danych internetowych.

## Opis projektu

Projekt przedstawia analizę danych hotelowych pochodzących z serwisu TripAdvisor
z wykorzystaniem metod machine learning oraz NLP.

Dane zostały pobrane i przetworzone wcześniej. Aplikacja FastAPI działa offline
na zapisanych plikach CSV oraz artefaktach modeli i nie wymaga klucza zewnętrznego API.

Projekt obejmuje:

- clustering hoteli,
- system rekomendacji i wyszukiwania hoteli,
- klasyfikację sentymentu opinii,
- proste API udostępniające zapisane modele.

## Dane

Przetworzone dane wykorzystywane przez projekt znajdują się w `data/full/`:

- `hotels.csv` - dane 135 hoteli,
- `hotels_with_clusters.csv` - dane hoteli z przypisanymi klastrami,
- `reviews.csv` - 468 opinii hotelowych.

Notebook `clear_data.ipynb` przedstawia etap wyboru kolumn, czyszczenia danych
oraz przygotowania plików używanych przez modele.

## Główne moduły

### Clustering hoteli

Notebook: `grupowanie.ipynb`

Hotele są grupowane na podstawie ocen, popularności, rankingu, poziomu cen,
udziału dobrych i złych ocen oraz szczegółowych ocen jakości.

Etapy:

- feature engineering,
- uzupełnianie braków medianami,
- standaryzacja cech,
- porównanie liczby klastrów za pomocą kilku metryk,
- KMeans,
- wizualizacja PCA,
- zapis modelu, scalera, median i listy cech.

Wybrano `k=4`. Wartość ta uzyskała najlepszy wynik Davies-Bouldin i pozwoliła
wydzielić cztery interpretowalne grupy hoteli:

1. najlepsze i najpopularniejsze hotele,
2. hotele przeciętne lub ekonomiczne,
3. problematyczne hotele,
4. dobrze oceniane, ale słabo zweryfikowane hotele.

### System rekomendacji

Notebook: `rekomendacje.ipynb`

Indeks TF-IDF tworzony jest na podstawie:

- nazwy hotelu,
- miasta,
- opisu,
- udogodnień,
- stylów hotelu.

Podobieństwo hoteli oraz dopasowanie zapytań tekstowych obliczane jest za pomocą
cosine similarity. Zapisany indeks zawiera 135 hoteli i 9234 cechy tekstowe.

### Klasyfikacja opinii

Notebook: `klasyfikacja_opinii.ipynb`

Sentyment jest definiowany jako klasyfikacja binarna:

- oceny 4-5: opinia pozytywna,
- oceny 1-3: opinia negatywna.

Tekst wejściowy powstaje z połączenia tytułu oraz treści opinii. Dane są
najpierw dzielone na zbiór treningowy i testowy, a TF-IDF jest dopasowywany
wyłącznie na zbiorze treningowym.

Zastosowany model: Logistic Regression z wagami klas.

Wyniki na zbiorze testowym:

- accuracy: `0.8723`,
- macro F1-score: `0.75`,
- recall klasy negatywnej: `0.42`,
- recall klasy pozytywnej: `0.99`.

## Artefakty modeli

Zapisane modele i indeksy znajdują się w `models/`:

- `models/clustering/` - KMeans, scaler, mediany i lista cech,
- `models/tf_idf/tfidf_index.joblib` - indeks systemu rekomendacji,
- `models/classification/sentiment_model.joblib` - vectorizer i model sentymentu.

## API

FastAPI udostępnia modele działające wyłącznie na lokalnych danych i artefaktach.

| Metoda | Endpoint | Opis |
|---|---|---|
| `GET` | `/` | Sprawdzenie działania API |
| `GET` | `/clusters` | Lista klastrów z podsumowaniami |
| `GET` | `/clusters/{cluster_id}` | Hotele należące do wybranego klastra |
| `POST` | `/predict-cluster` | Predykcja klastra dla nowych danych hotelu |
| `GET` | `/recommend/{location_id}` | Rekomendacje podobnych hoteli |
| `GET` | `/search-hotels` | Wyszukiwanie hoteli zapytaniem tekstowym |
| `POST` | `/predict-sentiment` | Klasyfikacja sentymentu opinii |

## Uruchomienie

Instalacja zależności:

```bash
pip install -r requirements.txt
```

Uruchomienie API z katalogu głównego projektu:

```bash
uvicorn api.app:app --reload
```

Dokumentacja i możliwość testowania endpointów:

```text
http://127.0.0.1:8000/docs
```

Przykładowe zapytania:

```text
GET /clusters
GET /recommend/278399?limit=3
GET /search-hotels?query=free%20internet&limit=3
```

```json
POST /predict-sentiment
{
  "review": "Great clean hotel and helpful staff"
}
```

## Technologie

- Python
- FastAPI
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Status

Projekt ukończony. Wszystkie endpointy działają offline na zapisanych danych
i artefaktach modeli.
