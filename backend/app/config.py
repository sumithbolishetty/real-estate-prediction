from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts/model/price_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts/preprocessing/metadata.json"
FRONTEND_ORIGIN = "http://localhost:3100"
FRONTEND_ORIGINS = [FRONTEND_ORIGIN, "http://127.0.0.1:3100"]


def load_metadata():
    if not METADATA_PATH.is_file():
        raise RuntimeError(f"Model metadata is missing at {METADATA_PATH}. Train the model from ml/ first.")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    for key in ("locations", "property_types", "parking_options", "amenities", "feature_columns"):
        if not metadata.get(key):
            raise RuntimeError(f"Model metadata has no supported {key}.")
    return metadata
