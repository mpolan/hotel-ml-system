from pathlib import Path

CLUSTER_INFO = {
    0: "Najlepsze i najpopularniejsze hotele",
    1: "Hotele przeciętne / ekonomiczne",
    2: "Problematyczne hotele",
    3: "Hotele bez wystarczającej liczby opinii",
}

BASE_DIR = Path(__file__).resolve().parents[1]
HOTELS_CSV = BASE_DIR / "data" / "full" / "hotels.csv"
CLUSTERS_CSV = BASE_DIR / "data" / "full" / "hotels_with_clusters.csv"

MODEL_DIR = BASE_DIR / "models" / "clustering"

KMEANS_MODEL_PATH = MODEL_DIR / "kmeans_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
FEATURE_MEDIANS_PATH = MODEL_DIR / "feature_medians.joblib"
FEATURES_FINAL_PATH = MODEL_DIR / "features_final.json"

TFIDF_INDEX_PATH = BASE_DIR / "models" / "tf_idf" / "tfidf_index.joblib"

SENTIMENT_MODEL_PATH = BASE_DIR / "models" / "classification" / "sentiment_model.joblib"
