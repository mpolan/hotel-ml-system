import pandas as pd
import requests

from tripadvisor_config import API_KEY, BASE_URL, RAW_DIR


MIASTA = [
    "Bialystok",
    "Warszawa",
    "Krakow",
    "Katowice",
    "Poznan",
    "Gdynia",
    "Gdansk",
    "Sopot",
    "Torun",
    "Bydgoszcz",
    "Wroclaw",
    "Suwalki",
    "Zakopane",
    "Kalisz",
    "Pisz",
]

context = []

for miasto in MIASTA:
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
    data = response_json.get("data", response_json)

    if not data:
        print(f"Brak danych dla {miasto}: {response_json}")
        continue

    for item in data:
        item["query_city"] = miasto

    context.extend(data)

df = pd.json_normalize(context)
df.to_csv(RAW_DIR / "hotele_raw.csv", index=False, encoding="utf-8-sig")
