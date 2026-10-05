import sys
from functools import lru_cache
import joblib
import pandas as pd
from .config import MODEL_PATH, load_metadata

ML_DIR = MODEL_PATH.parents[2] / "ml"
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
from preprocessing import FEATURE_COLUMNS, normalize_category, parse_amenities, to_model_frame


@lru_cache(maxsize=1)
def get_predictor():
    metadata = load_metadata()
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model artifact is missing at {MODEL_PATH}. Run training from ml/.")
    model = joblib.load(MODEL_PATH)
    return model, metadata


def options():
    _, metadata = get_predictor()
    return {key: metadata[key] for key in ("locations", "property_types", "parking_options", "amenities", "model_name")}


def format_indian_currency(amount: int) -> str:
    digits = str(abs(amount))
    if len(digits) > 3:
        tail, head = digits[-3:], digits[:-3]
        groups = []
        while head:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join(groups + [tail])
    return ("-" if amount < 0 else "") + "₹" + digits


def predict(payload):
    model, metadata = get_predictor()
    supported = {"location": metadata["locations"], "property_type": metadata["property_types"],
                 "parking": metadata["parking_options"]}
    values = payload.model_dump()
    for key, categories in supported.items():
        values[key] = normalize_category(values[key])
        if values[key] not in categories:
            raise ValueError(f"Unsupported {key.replace('_', ' ')}. Choose one of the available options.")
    allowed_amenities = set(metadata["amenities"])
    amenities = parse_amenities(values["amenities"])
    unknown = set(amenities) - allowed_amenities
    if unknown:
        raise ValueError(f"Unsupported amenity: {', '.join(sorted(unknown))}")
    values["amenities"] = amenities
    frame = pd.DataFrame([{key: values[key] for key in FEATURE_COLUMNS}])
    estimate = float(model.predict(to_model_frame(frame))[0])
    if not (estimate > 0):
        raise RuntimeError("Model returned a non-positive estimate")
    rounded = round(estimate)
    return {"estimated_price": int(rounded), "formatted_price": format_indian_currency(rounded)}
