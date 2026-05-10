import requests
from fastapi import APIRouter

from tripadvisor_config import API_KEY, BASE_URL


router = APIRouter()


@router.get("/test")
def get_test_hotels():
    params = {
        "key": API_KEY,
        "searchQuery": "Warszawa",
        "category": "hotels",
        "language": "en",
    }

    response = requests.get(
        f"{BASE_URL}/location/search",
        params=params,
        headers={"accept": "application/json"},
    )
    response_json = response.json()
    return response_json.get("data", response_json)


@router.get("/review/{location_id}")
async def get_review(location_id):
    params = {
        "key": API_KEY,
        "language": "en",
    }

    response = requests.get(
        f"{BASE_URL}/location/{location_id}/reviews",
        params=params,
        headers={"accept": "application/json"},
    )

    response_json = response.json()
    return response_json.get("data", response_json)


@router.get("/detail/{location_id}")
async def get_detail(location_id):
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
    return response_json.get("data", response_json)


@router.get("/miasto/{miasto}")
def get_hotels_by_city(miasto):
    params = {
        "key": API_KEY,
        "searchQuery": miasto,
        "category": "hotels",
        "language": "en",
    }

    response = requests.get(
        f"{BASE_URL}/location/search",
        params=params,
        headers={"accept": "application/json"},
    )
    response_json = response.json()
    return response_json.get("data", response_json)
