from pathlib import Path
import joblib
import pandas as pd
from preprocessing import FEATURE_COLUMNS, AMENITY_VALUES, parse_amenities, to_model_frame

ROOT = Path(__file__).resolve().parents[1]
model_path = ROOT / "artifacts/model/price_pipeline.joblib"
if not model_path.is_file():
    raise SystemExit("Model artifact not found. From ml/, run python train.py first.")
pipeline = joblib.load(model_path)
sample = pd.DataFrame([{"location": "pune", "area": 1200, "bedrooms": 3, "bathrooms": 2,
                        "property_type": "apartment", "parking": "yes", "property_age": 5,
                        "amenities": parse_amenities(["Gym", "Lift", "Security"])}])
print(float(pipeline.predict(to_model_frame(sample[FEATURE_COLUMNS]))[0]))
