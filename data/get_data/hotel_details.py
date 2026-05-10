import pandas as pd
import requests

from tripadvisor_config import API_KEY, BASE_URL, RAW_DIR


df = pd.read_csv(RAW_DIR / "hotele_raw.csv")
location_ids = df.get("location_id", [])

context = []

for location_id in location_ids:
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
    print(f"Znaleziono dane dla: {location_id}...")
    context.append(data)

df_full = pd.json_normalize(context)
df_full.to_csv(RAW_DIR / "details_raw.csv", index=False, encoding="utf-8-sig")
