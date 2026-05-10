import pandas as pd
import numpy as np
import joblib
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

from .config import CLUSTER_INFO, CLUSTERS_CSV, MODEL_DIR


app = FastAPI()


class HotelInput(BaseModel):
    name_details: str

    rating: float | None = None
    num_reviews: int | None = None
    price_level: str | None = None

    ranking: int | None = None
    ranking_out_of: int | None = None

    review_rating_count_1: int | None = None
    review_rating_count_2: int | None = None
    review_rating_count_3: int | None = None
    review_rating_count_4: int | None = None
    review_rating_count_5: int | None = None

    location_rating: float | None = None
    rooms_rating: float | None = None
    service_rating: float | None = None
    value_rating: float | None = None
    cleanliness_rating: float | None = None

PRICE_MAPPING = {
    "$": 1,
    "$$": 2,
    "$$$": 3,
    "$$$$": 4
}

hotels = pd.read_csv('data/full/hotels.csv')
print(hotels.columns)

@app.get("/")
def root():
    return {"message": "API dziala"}

@app.get("/clusters")
def get_clusters(limit: int = 5):
    df = pd.read_csv(CLUSTERS_CSV)
    df = df.replace({np.nan: None})

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
            .to_dict(orient="records")
        )

        result.append({
            "cluster": int(cluster_id),
            "cluster_info": CLUSTER_INFO[int(cluster_id)],
            "count": int(len(group)),
            "summary": summary,
            "examples": examples
        })

    return result

@app.get("/clusters/{cluster_id}")
def get_all_cluster(cluster_id: int):
    df = pd.read_csv(CLUSTERS_CSV)
    df = df.replace({np.nan: None})

    group = df[df["cluster"] == cluster_id]
    if group.empty:
        return {
            "error": "Nie znaleziono clustera",
            "cluster": cluster_id
        }

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

    hotels = group[hotel_cols].to_dict(orient="records")
    
    context = {
        "cluster": cluster_id,
        "cluster_info": CLUSTER_INFO[cluster_id],
        "count": int(len(group)),
        "summary": summary,
        "hotels": hotels,
    }

    return context


model = joblib.load(MODEL_DIR / "kmeans_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_medians = joblib.load(MODEL_DIR / "feature_medians.joblib")

with open(MODEL_DIR / "features_final.json", encoding="utf-8") as f:
    features_final = json.load(f)

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

    subratings_confidence = sum(x is not None for x in subratings) / len(subratings)

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

        "bad_review_share": safe_div(
            bad_reviews,
            num_reviews
        ),

        "excellent_review_share": safe_div(
            excellent_reviews,
            num_reviews
        ),

        "location_rating": data["location_rating"],
        "rooms_rating": data["rooms_rating"],
        "service_rating": data["service_rating"],
        "value_rating": data["value_rating"],
        "cleanliness_rating": data["cleanliness_rating"],

        "subratings_confidence": subratings_confidence,
    }

    row = {}

    for feature in features_final:
        value = engineered_data.get(feature)

        if value is None or pd.isna(value):
            value = feature_medians[feature]

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
    }

from data.get_data.tripadvisor_config import BASE_URL, API_KEY
import requests

def fetch_hotels_search(city: str) -> list[dict]:
    params = {
        "key": API_KEY,
        "searchQuery": city,
        "category": "hotels",
        "language": "en",
    }

    response = requests.get(
        f"{BASE_URL}/location/search",
        params=params,
        headers={"accept": "application/json"},
    )

    response_json = response.json()
    data = response_json.get("data", response_json)

    if not data:
        print(f"Brak danych dla {city}: {response_json}")

    for item in data:
        item["query_city"] = city

    return data

def fetch_hotel_details(location_id: str) -> dict:
    params = {
        "key": API_KEY,
        "language": "en",
        "currency": "USD",
    }

    response = requests.get(
        f"{BASE_URL}/location/{location_id}/details",
        params=params,
        headers={"accept": "application/json"},
    )

    response_json = response.json()
    data = response_json.get("data", response_json)

    return data

def merge_hotel_data(search_item: dict, details: dict, city: str) -> dict:
    return {
        "search": search_item,
        "details": details,
        "query_city": city,
    }

@app.get("/get_hotels/{city}")
def get_hotels_from_city(city: str, limit: int = 10):
    search_data = fetch_hotels_search(city)

    hotels = []

    for item in search_data[:limit]:
        location_id = item.get("location_id")

        if not location_id:
            continue

        details = fetch_hotel_details(location_id)

        hotels.append(
            merge_hotel_data(item, details, city)
        )

    return {
        "city": city,
        "count": len(hotels),
        "hotels": hotels,
    }

    
