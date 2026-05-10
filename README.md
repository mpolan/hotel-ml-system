# Hotel Analytics ML

Projekt realizowany w ramach kursu eksploracji danych internetowych.

## Opis projektu

Projekt skupia się na analizie danych hotelowych pochodzących z serwisu TripAdvisor z wykorzystaniem metod machine learning oraz NLP.

Aktualnie projekt obejmuje:

- clustering hoteli,
- system rekomendacji hoteli,
- klasyfikację opinii użytkowników.

## Technologie

- Python
- FastAPI
- pandas
- scikit-learn
- numpy
- matplotlib
- nltk

## Główne moduły

### Clustering hoteli
Grupowanie hoteli na podstawie:
- ocen,
- popularności,
- rankingów,
- subratings,
- jakości opinii.

Algorytmy:
- KMeans
- PCA

---

### System rekomendacji
Rekomendowanie podobnych hoteli na podstawie:
- opisów,
- opinii,
- cech tekstowych.

Algorytmy:
- TF-IDF
- cosine similarity

---

### Klasyfikacja opinii
Klasyfikacja sentymentu opinii hotelowych.

Algorytmy:
- Logistic Regression
- TF-IDF

## API

Projekt wykorzystuje FastAPI do udostępniania endpointów ML oraz integracji z danymi TripAdvisor API.

## Uruchomienie projektu

### Instalacja zależności

```bash
pip install -r requirements.txt
```

### Uruchomienie API:
```bash
uvicorn api.app:app --reload
```

### Dokumentacja API:
```bash
http://127.0.0.1:8000/docs
```

## Status

Projekt w trakcie rozwoju.