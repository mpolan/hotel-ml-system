import pandas as pd
import numpy as np
import joblib
import json
from fastapi import FastAPI, Path, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, StringConstraints
from typing import Annotated, Any
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    CLUSTER_INFO,
    CLUSTERS_CSV,
    FEATURE_MEDIANS_PATH,
    FEATURES_FINAL_PATH,
    KMEANS_MODEL_PATH,
    SCALER_PATH,
    SENTIMENT_MODEL_PATH,
    TFIDF_INDEX_PATH,
)


app = FastAPI()

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Rating = Annotated[float, Field(ge=0, le=5)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class HotelInput(BaseModel):
    name_details: NonEmptyStr

    rating: Rating | None = None
    num_reviews: NonNegativeInt | None = None
    price_level: str | None = None

    ranking: Annotated[int, Field(ge=1)] | None = None
    ranking_out_of: Annotated[int, Field(ge=1)] | None = None

    review_rating_count_1: NonNegativeInt | None = None
    review_rating_count_2: NonNegativeInt | None = None
    review_rating_count_3: NonNegativeInt | None = None
    review_rating_count_4: NonNegativeInt | None = None
    review_rating_count_5: NonNegativeInt | None = None

    location_rating: Rating | None = None
    rooms_rating: Rating | None = None
    service_rating: Rating | None = None
    value_rating: Rating | None = None
    cleanliness_rating: Rating | None = None

PRICE_MAPPING = {
    "$": 1,
    "$$": 2,
    "$$$": 3,
    "$$$$": 4
}

class SentimentInput(BaseModel):
    review: NonEmptyStr

@app.get("/")
def root():
    return {"message": "API dziala"}


@app.get("/ui", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html lang="pl">
<head>
    <meta charset="utf-8">
    <title>Hotel Analytics ML</title>
</head>
<body>
    <h1>Hotel Analytics ML</h1>
    <p><a href="/docs">FastAPI docs</a></p>

    <hr>

    <h2>Klastry</h2>
    <button type="button" onclick="loadClusters()">Pokaż klastry</button>

    <form id="cluster-details-form">
        <p>
            <label>
                ID klastra:
                <input name="cluster_id" type="number" min="0" value="0" required>
            </label>
            <button type="submit">Pokaż hotele klastra</button>
        </p>
    </form>

    <hr>

    <h2>Rekomendacje hotelu</h2>
    <form id="recommend-form">
        <p>
            <label>
                Location ID:
                <input name="location_id" type="number" min="1" value="278399" required>
            </label>
        </p>
        <p>
            <label>
                Limit:
                <input name="limit" type="number" min="1" value="5" required>
            </label>
        </p>
        <button type="submit">Pokaż rekomendacje</button>
    </form>

    <hr>

    <h2>Wyszukiwanie hoteli</h2>
    <form id="search-form">
        <p>
            <label>
                Zapytanie:
                <input name="query" value="free internet" required>
            </label>
        </p>
        <p>
            <label>
                Limit:
                <input name="limit" type="number" min="1" value="5" required>
            </label>
        </p>
        <button type="submit">Szukaj</button>
    </form>

    <hr>

    <h2>Klasyfikacja opinii</h2>
    <form id="sentiment-form">
        <p>
            <label>
                Opinia:
                <textarea name="review" rows="4" cols="60" required>Great clean hotel and helpful staff</textarea>
            </label>
        </p>
        <button type="submit">Sprawdź sentyment</button>
    </form>

    <hr>

    <h2>Predykcja klastra hotelu</h2>
    <form id="predict-cluster-form">
        <p><label>Nazwa: <input name="name_details" value="Test Hotel" required></label></p>

        <p><label>Ocena 0-5:
            <input name="rating" type="number" min="0" max="5" step="0.1" value="4.4">
        </label></p>

        <p><label>Liczba opinii:
            <input name="num_reviews" type="number" min="0" value="150">
        </label></p>
        <p>
            <label>
                Poziom cen:
                <select name="price_level">
                    <option value="">Brak danych</option>
                    <option value="$">$</option>
                    <option value="$$">$$</option>
                    <option value="$$$">$$$</option>
                    <option value="$$$$">$$$$</option>
                </select>
            </label>
        </p>
        <p><label>Pozycja w rankingu:
            <input name="ranking" type="number" min="1" value="10">
        </label></p>

        <p><label>Liczba hoteli w rankingu:
            <input name="ranking_out_of" type="number" min="1" value="100">
        </label></p>
        <p><label>Location rating:
            <input name="location_rating" type="number" min="0" max="5" step="0.1" value="4.5">
        </label></p>

        <p><label>Rooms rating:
            <input name="rooms_rating" type="number" min="0" max="5" step="0.1" value="4.4">
        </label></p>

        <p><label>Service rating:
            <input name="service_rating" type="number" min="0" max="5" step="0.1" value="4.5">
        </label></p>

        <p><label>Value rating:
            <input name="value_rating" type="number" min="0" max="5" step="0.1" value="4.3">
        </label></p>

        <p><label>Cleanliness rating:
            <input name="cleanliness_rating" type="number" min="0" max="5" step="0.1" value="4.6">
        </label></p>
        <button type="submit">Przewidź klaster</button>
    </form>

    <hr>

    <h2>Wynik</h2>
    <pre id="result">Wybierz operację.</pre>

    <script>
        const result = document.getElementById("result");

        function show(data) {
            result.textContent = JSON.stringify(data, null, 2);
        }

        async function request(url, options) {
            try {
                const response = await fetch(url, options);
                const data = await response.json();
                show(data);
            } catch (error) {
                show({error: String(error)});
            }
        }

        function optionalNumber(formData, name) {
            const value = formData.get(name);
            return value === "" ? null : Number(value);
        }

        function loadClusters() {
            request("/clusters");
        }

        document.getElementById("cluster-details-form").addEventListener("submit", event => {
            event.preventDefault();
            const data = new FormData(event.target);
            request(`/clusters/${data.get("cluster_id")}`);
        });

        document.getElementById("recommend-form").addEventListener("submit", event => {
            event.preventDefault();
            const data = new FormData(event.target);
            request(`/recommend/${data.get("location_id")}?limit=${data.get("limit")}`);
        });

        document.getElementById("search-form").addEventListener("submit", event => {
            event.preventDefault();
            const data = new FormData(event.target);
            const query = encodeURIComponent(data.get("query"));
            request(`/search-hotels?query=${query}&limit=${data.get("limit")}`);
        });

        document.getElementById("sentiment-form").addEventListener("submit", event => {
            event.preventDefault();
            const data = new FormData(event.target);
            request("/predict-sentiment", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({review: data.get("review")})
            });
        });

        document.getElementById("predict-cluster-form").addEventListener("submit", event => {
            event.preventDefault();
            const data = new FormData(event.target);
            request("/predict-cluster", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({
                name_details: data.get("name_details"),
                rating: optionalNumber(data, "rating"),
                num_reviews: optionalNumber(data, "num_reviews"),
                price_level: data.get("price_level") || null,
                ranking: optionalNumber(data, "ranking"),
                ranking_out_of: optionalNumber(data, "ranking_out_of"),
                location_rating: optionalNumber(data, "location_rating"),
                rooms_rating: optionalNumber(data, "rooms_rating"),
                service_rating: optionalNumber(data, "service_rating"),
                value_rating: optionalNumber(data, "value_rating"),
                cleanliness_rating: optionalNumber(data, "cleanliness_rating")
            })
            });
        });
    </script>
</body>
</html>
"""


def records_with_none(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.astype(object).where(pd.notna(df), None).to_dict(orient="records")


@app.get("/clusters")
def get_clusters(limit: Annotated[int, Query(ge=1)] = 5):
    df = pd.read_csv(CLUSTERS_CSV)

    result = []

    summary_cols = [
        "rating",
        "num_reviews",
        "log_num_reviews",
        "ranking_ratio",
        "bad_review_share",
        "excellent_review_share",
        "subratings_confidence",
    ]

    example_cols = [
        "location_id",
        "address_obj.city_details",
        "name_details",
        "rating",
        "num_reviews",
        "log_num_reviews",
        "ranking_ratio",
        "bad_review_share",
        "excellent_review_share",
        "price_level",
    ]

    for cluster_id, group in df.groupby("cluster"):
        summary = (
            group[summary_cols]
            .mean(numeric_only=True)
            .round(4)
            .to_dict()
        )

        examples = (
            group[example_cols]
            .head(limit)
        )
        examples = records_with_none(examples)

        result.append({
            "cluster": int(cluster_id),
            "cluster_info": CLUSTER_INFO[int(cluster_id)],
            "count": int(len(group)),
            "summary": summary,
            "examples": examples
        })

    return result

@app.get("/clusters/{cluster_id}")
def get_all_cluster(cluster_id: Annotated[int, Path(ge=0)]):
    df = pd.read_csv(CLUSTERS_CSV)

    group = df[df["cluster"] == cluster_id]
    if group.empty:
        return {
            "error": "Nie znaleziono clustera",
            "cluster": cluster_id
        }

    summary_cols = [
        "rating",
        "num_reviews",
        "log_num_reviews",
        "ranking_ratio",
        "bad_review_share",
        "excellent_review_share",
        "subratings_confidence",
    ]

    hotel_cols = [
        "location_id",
        "address_obj.city_details",
        "name_details",
        "rating",
        "num_reviews",
        "log_num_reviews",
        "ranking_ratio",
        "bad_review_share",
        "excellent_review_share",
        "price_level",
    ]

    summary = (
        group[summary_cols]
        .mean(numeric_only=True)
        .round(4)
        .to_dict()
    )

    hotels = records_with_none(group[hotel_cols])
    
    context = {
        "cluster": cluster_id,
        "cluster_info": CLUSTER_INFO[cluster_id],
        "count": int(len(group)),
        "summary": summary,
        "hotels": hotels,
    }

    return context


model = joblib.load(KMEANS_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_medians = joblib.load(FEATURE_MEDIANS_PATH)

with open(FEATURES_FINAL_PATH, encoding="utf-8") as f:
    features_final = json.load(f)

#*====================== TF-IDF ==============================

tfidf_index = joblib.load(TFIDF_INDEX_PATH)
tfidf_vectorizer = tfidf_index['vectorizer']
tfidf_matrix = tfidf_index['matrix']
tfidf_hotels = tfidf_index['hotels']


#*================================ CLASSIFICATION: ================================



sentiment_data = joblib.load(SENTIMENT_MODEL_PATH)

sentiment_vectorizer = sentiment_data["vectorizer"]
sentiment_model = sentiment_data["model"]

def safe_div(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b

@app.post("/predict-cluster")
def predict_cluster(hotel: HotelInput):
    data = hotel.model_dump()

    num_reviews = data["num_reviews"]

    bad_reviews = None
    excellent_reviews = None

    if data["review_rating_count_1"] is not None and data["review_rating_count_2"] is not None:
        bad_reviews = data["review_rating_count_1"] + data["review_rating_count_2"]

    if data["review_rating_count_5"] is not None:
        excellent_reviews = data["review_rating_count_5"]

    subratings = [
        data["location_rating"],
        data["rooms_rating"],
        data["service_rating"],
        data["value_rating"],
        data["cleanliness_rating"],
    ]

    n_subratings = sum(x is not None for x in subratings)

    ALPHA = 5
    subratings_confidence = n_subratings / (n_subratings + ALPHA)


    if num_reviews == 0:
        bad_review_share = 0
        excellent_review_share = 0
    else:
        bad_review_share = safe_div(bad_reviews, num_reviews)
        excellent_review_share = safe_div(excellent_reviews, num_reviews)


    engineered_data = {
        "rating": data["rating"],

        "log_num_reviews": (
            np.log1p(num_reviews)
            if num_reviews is not None
            else None
        ),

        "ranking_ratio": safe_div(
            data["ranking"],
            data["ranking_out_of"]
        ),

        "price_level_num": (
            PRICE_MAPPING.get(data["price_level"])
            if data["price_level"] is not None
            else None
        ),

        "bad_review_share": bad_review_share,
        "excellent_review_share": excellent_review_share,

        "location_rating": data["location_rating"],
        "rooms_rating": data["rooms_rating"],
        "service_rating": data["service_rating"],
        "value_rating": data["value_rating"],
        "cleanliness_rating": data["cleanliness_rating"],

        "subratings_confidence": subratings_confidence,
    }

    row = {}
    imputed_features = []

    for feature in features_final:
        value = engineered_data.get(feature)

        if value is None or pd.isna(value):
            value = feature_medians[feature]
            imputed_features.append(feature)

        row[feature] = value


    df_new = pd.DataFrame([row])
    df_new = df_new[features_final]

    X_scaled = scaler.transform(df_new)

    cluster_id = int(model.predict(X_scaled)[0])

    return {
        "hotel": hotel.name_details,
        "cluster": cluster_id,
        "cluster_info": CLUSTER_INFO.get(cluster_id) or CLUSTER_INFO.get(str(cluster_id)),
        "used_features": row,
        "imputed_features": imputed_features,
        "imputed_count": len(imputed_features),
        "prediction_warning": (
            "Predykcja oparta częściowo na medianach treningowych"
            if imputed_features
            else None
        )
    }

@app.get("/recommend/{location_id}")
def recommend_hotels(
    location_id: Annotated[int, Path(ge=1)],
    limit: Annotated[int, Query(ge=1)] = 5,
):
    hotel_idx = None

    for idx, hotel in enumerate(tfidf_hotels):
        if int(hotel['location_id']) == location_id:
            hotel_idx = idx
            break

    if hotel_idx is None:
        return{
            "error": "Nie znaleziono hotelu",
            "location_id": location_id
        }

    scores = cosine_similarity(
        tfidf_matrix[hotel_idx], 
        tfidf_matrix
    ).flatten()

    best = scores.argsort()[::-1]

    recommendations = []

    for i in best:
        if i == hotel_idx:
            continue
        hotel = tfidf_hotels[i]

        recommendations.append({
            "location_id": int(hotel["location_id"]),
            "name": hotel.get("name_details"),
            "city": hotel.get("address_obj.city_details"),
            "rating": hotel.get("rating"),
            "price_level": hotel.get("price_level"),
            "similarity_score": round(float(scores[i]), 4),
        })

        if len(recommendations) >= limit:
            break

    selected_hotel = tfidf_hotels[hotel_idx]
    context = {
        "source_hotel": {
            "location_id": int(selected_hotel["location_id"]),
            "name": selected_hotel.get("name_details"),
            "city": selected_hotel.get("address_obj.city_details"),
        },
        "recommendations": recommendations
    }

    return context

@app.get("/search-hotels")
def search_hotels(
    query: str,
    limit: Annotated[int, Query(ge=1)] = 5,
):
    q = tfidf_vectorizer.transform([query])

    if q.nnz == 0:
        return {
            "query": query,
            "results": [],
        }

    scores = cosine_similarity(q, tfidf_matrix).flatten()
    best_indices = scores.argsort()[::-1][:limit]

    results = []

    for idx in best_indices:
        hotel = tfidf_hotels[idx]

        results.append({
            "location_id": int(hotel["location_id"]),
            "name": hotel.get("name_details"),
            "city": hotel.get("address_obj.city_details"),
            "rating": hotel.get("rating"),
            "price_level": hotel.get("price_level"),
            "similarity_score": round(float(scores[idx]), 4),
        })

    return {
        "query": query,
        "results": results
    }


@app.post("/predict-sentiment")
def predict_sentiment(input_data: SentimentInput):
    text = input_data.review

    X = sentiment_vectorizer.transform([text])

    prediction = int(sentiment_model.predict(X)[0])

    result = "positive" if prediction == 1 else "negative"

    response = {
        "review": text,
        "sentiment": result,
        "label": prediction,
    }

    if hasattr(sentiment_model, "predict_proba"):
        probabilities = sentiment_model.predict_proba(X)[0]

        response["probability"] = {
            "negative": round(float(probabilities[0]), 4),
            "positive": round(float(probabilities[1]), 4),
        }

    return response
