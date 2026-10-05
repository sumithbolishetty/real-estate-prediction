"""Shared, deterministic validation and feature definitions for training/inference."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

FEATURE_COLUMNS = ["location", "area", "bedrooms", "bathrooms", "property_type", "parking", "property_age", "amenities"]
TARGET = "price"
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "property_age"]
CATEGORICAL_FEATURES = ["location", "property_type", "parking"]
AMENITY_VALUES = ["Gym", "Swimming Pool", "Garden", "Security", "Lift", "Club House", "Power Backup", "Parking"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + [TARGET]


def normalize_category(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def parse_amenities(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        values = value
    elif pd.isna(value) or str(value).strip() == "":
        values = []
    else:
        values = re.split(r"[|,;]", str(value))
    normalized = {normalize_category(item) for item in values if str(item).strip()}
    lookup = {normalize_category(item): item for item in AMENITY_VALUES}
    unknown = normalized - lookup.keys()
    if unknown:
        raise ValueError(f"Unsupported amenities: {', '.join(sorted(unknown))}")
    return sorted((lookup[item] for item in normalized), key=AMENITY_VALUES.index)


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop only records invalid for the model; keep valid outliers and report them."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    source_rows = len(frame)
    source_categorical_values = {
        col: sorted(frame[col].dropna().astype(str).unique().tolist())
        for col in CATEGORICAL_FEATURES
    }
    source_amenity_bundles = sorted(frame["amenities"].dropna().astype(str).unique().tolist())
    source_amenity_tokens = sorted({
        token.strip()
        for bundle in source_amenity_bundles
        for token in re.split(r"[|,;]", bundle)
        if token.strip()
    })
    category_collisions = {}
    for col in CATEGORICAL_FEATURES:
        groups = {}
        for value in source_categorical_values[col]:
            groups.setdefault(normalize_category(value), []).append(value)
        category_collisions[col] = {key: values for key, values in groups.items() if len(values) > 1}
    df = frame[REQUIRED_COLUMNS].copy()
    df = df.drop_duplicates()
    duplicate_rows = source_rows - len(df)
    missing_rows = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    df = df.dropna(subset=REQUIRED_COLUMNS)
    blank_category_mask = df[CATEGORICAL_FEATURES].astype(str).apply(
        lambda column: column.str.strip().eq("")
    ).any(axis=1)
    blank_category_rows = int(blank_category_mask.sum())
    df = df.loc[~blank_category_mask].copy()
    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    coerced_missing = int(df[NUMERIC_FEATURES + [TARGET]].isna().any(axis=1).sum())
    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET])
    finite = np.isfinite(df[NUMERIC_FEATURES + [TARGET]]).all(axis=1)
    invalid = (~finite | (df.area <= 0) | (df.bedrooms < 1) | (df.bathrooms <= 0) |
               (df.property_age < 0) | (df.price <= 0))
    invalid_rows = int(invalid.sum())
    df = df.loc[~invalid].copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].map(normalize_category)
    df["amenities"] = df["amenities"].map(parse_amenities)
    if len(df) < 8:
        raise ValueError(f"At least 8 valid rows are required after cleaning; got {len(df)}")
    numeric_summary = {}
    for col in NUMERIC_FEATURES + [TARGET]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        numeric_summary[col] = {"min": float(df[col].min()), "max": float(df[col].max()),
                                "iqr_outlier_count": int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())}
    report = {"source_rows": source_rows, "rows_after_cleaning": len(df), "columns": list(frame.columns),
              "data_types": {key: str(value) for key, value in frame.dtypes.items()},
              "raw_categorical_values": source_categorical_values,
              "category_normalization_collisions": category_collisions,
              "amenity_source_bundle_count": len(source_amenity_bundles),
              "amenity_source_tokens": source_amenity_tokens,
              "duplicate_rows_removed": duplicate_rows, "rows_with_missing_values_removed": missing_rows,
              "rows_with_blank_categories_removed": blank_category_rows,
              "rows_with_invalid_numeric_values_removed": coerced_missing, "rows_with_invalid_values_removed": invalid_rows,
              "numerical_ranges_and_iqr_outlier_counts": numeric_summary,
              "locations": sorted(df.location.unique().tolist()),
              "property_types": sorted(df.property_type.unique().tolist()),
              "parking_options": sorted(df.parking.unique().tolist()),
              "amenities": sorted({a for row in df.amenities for a in row}, key=AMENITY_VALUES.index),
              "missing_values_before_cleaning": {k: int(v) for k, v in frame[REQUIRED_COLUMNS].isna().sum().items()},
              "categorical_normalization": "trim, case-fold, and collapse whitespace; amenities canonicalized against allowlist",
              "cleaning_policy": "remove exact duplicates, rows missing required fields, nonnumeric numeric values, and rows where area<=0, bedrooms<1, bathrooms<=0, property_age<0, or price<=0; retain valid outliers and report IQR counts"}
    return df.reset_index(drop=True), report


def to_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[FEATURE_COLUMNS].copy()
    result["amenities"] = result["amenities"].map(lambda items: "|".join(parse_amenities(items)))
    return result


def amenities_to_multihot(values):
    """Picklable transformer function shared by training and the saved pipeline."""
    matrix = values.to_numpy() if hasattr(values, "to_numpy") else values
    return np.asarray([[int(amenity in str(value).split("|")) for amenity in AMENITY_VALUES]
                       for value in matrix.reshape(-1)], dtype=float)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
