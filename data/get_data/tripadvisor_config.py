import os
from pathlib import Path

from dotenv import load_dotenv


DATA_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = DATA_DIR.parent
RAW_DIR = DATA_DIR / "raw"

os.makedirs(RAW_DIR, exist_ok=True)

load_dotenv(PROJECT_DIR / ".env")

API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("Brak klucza. Ustaw zmienna API_KEY w pliku .env.")

BASE_URL = "https://api.content.tripadvisor.com/api/v1"
