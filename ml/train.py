"""Evaluate models with shuffled 5-fold CV, then refit the selected pipeline on all valid rows."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from preprocessing import (
    AMENITY_VALUES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
    amenities_to_multihot,
    clean_dataset,
    to_model_frame,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/sample_properties.csv"
MODEL_DIR = ROOT / "artifacts/model"
PREPROCESSING_DIR = ROOT / "artifacts/preprocessing"
REPORT_PATH = ROOT / "data/processed/dataset_report.json"
EVALUATION_PATH = ROOT / "data/processed/evaluation.json"
CV_FOLDS = 5
RANDOM_STATE = 42


def make_pipeline(estimator):
    amenities = Pipeline([("multi_hot", FunctionTransformer(amenities_to_multihot, validate=False))])
    preprocess = ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("amenities", amenities, ["amenities"]),
    ])
    return Pipeline([("preprocessing", preprocess), ("model", estimator)])


def _score(y_true, prediction):
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** 0.5),
        "R2": float(r2_score(y_true, prediction)),
    }


def _cross_validate(name, estimator, x, y, folds):
    fold_results = []
    for fold_number, (train_idx, test_idx) in enumerate(folds.split(x, y), start=1):
        candidate = clone(make_pipeline(estimator)) if name != "Mean Price Baseline" else clone(estimator)
        candidate.fit(x.iloc[train_idx], y.iloc[train_idx])
        prediction = candidate.predict(x.iloc[test_idx])
        fold_results.append({"fold": fold_number, "test_rows": len(test_idx), **_score(y.iloc[test_idx], prediction)})
    summaries = {}
    for metric in ("MAE", "RMSE", "R2"):
        values = [row[metric] for row in fold_results]
        summaries[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1))}
    return {"summary": summaries, "folds": fold_results}


def feature_importance_report(pipeline):
    """Return model-relative tree importances with original feature names/groups."""
    preprocessor = pipeline.named_steps["preprocessing"]
    importances = pipeline.named_steps["model"].feature_importances_
    details = []

    for feature, value in zip(NUMERIC_FEATURES, importances[preprocessor.output_indices_["numeric"]]):
        details.append({"feature": feature, "group": feature, "importance": float(value)})

    encoder = preprocessor.named_transformers_["categorical"]
    names = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    for feature, value in zip(names, importances[preprocessor.output_indices_["categorical"]]):
        group, category = next((col, feature[len(col) + 1:]) for col in CATEGORICAL_FEATURES if feature.startswith(col + "_"))
        details.append({"feature": f"{group} = {category}", "group": group, "importance": float(value)})

    for amenity, value in zip(AMENITY_VALUES, importances[preprocessor.output_indices_["amenities"]]):
        details.append({"feature": f"amenity: {amenity}", "group": "amenities", "importance": float(value)})

    group_totals = {}
    for item in details:
        group_totals[item["group"]] = group_totals.get(item["group"], 0.0) + item["importance"]
    details.sort(key=lambda item: item["importance"], reverse=True)
    return {
        "interpretation": "Tree feature importance describes model reliance, not causation; small-data rankings are unstable.",
        "by_feature_group": dict(sorted(group_totals.items(), key=lambda item: item[1], reverse=True)),
        "top_encoded_features": details[:15],
    }


def feature_analysis(frame):
    numeric_summary = {}
    for feature in NUMERIC_FEATURES:
        numeric_summary[feature] = {
            "unique_values": int(frame[feature].nunique()),
            "constant": bool(frame[feature].nunique() <= 1),
            "min": float(frame[feature].min()),
            "max": float(frame[feature].max()),
            "pearson_correlation_with_price_descriptive_only": float(frame[feature].corr(frame[TARGET])),
        }
    categorical_summary = {}
    for feature in CATEGORICAL_FEATURES:
        values = frame[feature].astype(str)
        categorical_summary[feature] = {
            "unique_values": int(values.nunique()),
            "constant": bool(values.nunique() <= 1),
            "categories": sorted(values.unique().tolist()),
        }
    tokens = [amenity for row in frame["amenities"] for amenity in row]
    categorical_summary["amenities"] = {
        "representation": "pipe-delimited source values canonicalized into 8 multi-hot amenity flags",
        "distinct_bundles": int(pd.Series(frame["amenities"].map(lambda row: "|".join(row))).nunique()),
        "unique_amenity_flags": len(AMENITY_VALUES),
        "token_counts": {amenity: int(tokens.count(amenity)) for amenity in AMENITY_VALUES},
    }
    outlier_analysis = {}
    for feature in NUMERIC_FEATURES + [TARGET]:
        q1, q3 = frame[feature].quantile([0.25, 0.75])
        iqr = q3 - q1
        flagged = frame.loc[
            (frame[feature] < q1 - 1.5 * iqr) | (frame[feature] > q3 + 1.5 * iqr),
            FEATURE_COLUMNS + [TARGET],
        ]
        outlier_analysis[feature] = {
            "iqr_flag_count": len(flagged),
            "rows_retained": flagged.to_dict(orient="records"),
            "decision": "All statistically flagged rows retained; no clear data-entry error found. IQR flags alone are not grounds for deletion.",
        }
    q1, q3 = frame[TARGET].quantile([0.25, 0.75])
    return {
        "numerical_features": numeric_summary,
        "categorical_features": categorical_summary,
        "iqr_outlier_analysis": outlier_analysis,
        "price_distribution": {
            "min": float(frame[TARGET].min()), "q1": float(q1), "median": float(frame[TARGET].median()),
            "mean": float(frame[TARGET].mean()), "q3": float(q3), "max": float(frame[TARGET].max()),
            "iqr_outlier_count": outlier_analysis[TARGET]["iqr_flag_count"],
            "possible_iqr_outliers_retained": outlier_analysis[TARGET]["rows_retained"],
            "outlier_decision": "Retained: no clear data-entry error; flagged records have premium property attributes. Data is demo data, so real-world legitimacy cannot be independently established.",
        },
        "notes": [
            "No input feature is constant in the 42 rows.",
            "Location has 7 categories, property type 3, and parking 2; cardinality is modest.",
            "The 14 source amenity bundles are decomposed into 8 multi-hot amenity indicators.",
            "Numeric correlations are descriptive associations and do not establish cause.",
        ],
    }


def main():
    raw = pd.read_csv(DATA)
    df, data_report = clean_dataset(raw)
    x = to_model_frame(df)
    y = df[TARGET]
    if len(df) != 42:
        print(f"Note: expected 42 source rows for the current dataset; found {len(df)} after cleaning.")

    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "Random Forest Regression": RandomForestRegressor(n_estimators=300, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1),
        "Extra Trees Regression": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=1),
        "Hist Gradient Boosting Regression": HistGradientBoostingRegressor(max_iter=150, min_samples_leaf=3, l2_regularization=1.0, random_state=RANDOM_STATE),
    }
    cv_results = {
        "Mean Price Baseline": _cross_validate("Mean Price Baseline", DummyRegressor(strategy="mean"), x, y, folds),
    }
    for name, estimator in candidates.items():
        cv_results[name] = _cross_validate(name, estimator, x, y, folds)

    selected_name = min(candidates, key=lambda name: cv_results[name]["summary"]["MAE"]["mean"])
    selected = make_pipeline(candidates[selected_name])
    selected.fit(x, y)
    importance = feature_importance_report(selected)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = MODEL_DIR / "price_pipeline.joblib"
    joblib.dump(selected, artifact_path)

    model_metadata = {
        "model_name": selected_name,
        "training_dataset": "data/raw/sample_properties.csv",
        "training_rows": len(df),
        "feature_names": FEATURE_COLUMNS,
        "target_name": TARGET,
        "training_date_utc": datetime.now(timezone.utc).date().isoformat(),
        "evaluation_method": f"{CV_FOLDS}-fold shuffled K-fold cross-validation, random_state={RANDOM_STATE}",
        "evaluation": cv_results,
        "selection_reason": "Lowest mean MAE across the regression candidates in the same five validation folds.",
        "preprocessing": "Numeric pass-through; one-hot encoding for location/property_type/parking with unknown-category ignore; pipe-delimited amenities canonicalized and multi-hot encoded in the saved pipeline.",
        "feature_importance": importance,
        "limitations": "Only 42 small demo records; cross-validation variation is high and estimates are uncertain. Not a production valuation model.",
    }
    write_json(MODEL_DIR / "model_metadata.json", model_metadata)
    preprocessing_metadata = {
        "model_name": selected_name,
        "feature_columns": FEATURE_COLUMNS,
        "target": TARGET,
        "locations": data_report["locations"],
        "property_types": data_report["property_types"],
        "parking_options": data_report["parking_options"],
        "amenities": data_report["amenities"],
        "amenity_allowlist": AMENITY_VALUES,
        "training_rows": len(df),
        "evaluation_method": model_metadata["evaluation_method"],
        "evaluation": cv_results,
    }
    write_json(PREPROCESSING_DIR / "metadata.json", preprocessing_metadata)

    data_report.update({
        "dataset_source": "Existing project demo dataset at data/raw/sample_properties.csv; no additional compatible dataset was present",
        "feature_names": FEATURE_COLUMNS,
        "target_variable": TARGET,
        "categorical_features": CATEGORICAL_FEATURES + ["amenities (multi-hot)"],
        "numerical_features": NUMERIC_FEATURES,
        "cleaning_performed": "Exact duplicates, missing required values, nonnumeric numeric values, and invalid area/bedroom/bathroom/age/price rows are removed; category text is normalized; malformed amenity tokens are rejected. No rows required removal in this dataset.",
        "feature_analysis": feature_analysis(df),
        "model_comparison": cv_results,
        "selected_model": selected_name,
        "feature_importance": importance,
        "limitations": "The dataset contains only 42 existing demo records. CV scores and feature importance have high uncertainty and do not support claims of production accuracy or real-market validity.",
    })
    write_json(REPORT_PATH, data_report)
    write_json(EVALUATION_PATH, {
        "selected_model": selected_name,
        "folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "selection_metric": "lowest mean cross-validation MAE among regression models",
        "models": cv_results,
        "feature_importance": importance,
        "uncertainty_note": "Only 42 records; fold scores vary substantially and R² is especially unstable on small test folds.",
    })

    # Reload the final all-data artifact and predict from a real row in the dataset.
    loaded = joblib.load(artifact_path)
    sample = x.iloc[[0]]
    sample_prediction = float(loaded.predict(sample)[0])
    if not np.isfinite(sample_prediction) or sample_prediction <= 0:
        raise RuntimeError("Saved pipeline failed actual-row sample prediction validation")
    print("Five-fold cross-validation (mean ± fold standard deviation)")
    print("-----------------------------------------------------------")
    for name, result in cv_results.items():
        summary = result["summary"]
        print(f"{name}: MAE {summary['MAE']['mean']:,.0f} ± {summary['MAE']['std']:,.0f}; "
              f"RMSE {summary['RMSE']['mean']:,.0f} ± {summary['RMSE']['std']:,.0f}; "
              f"R² {summary['R2']['mean']:.3f} ± {summary['R2']['std']:.3f}")
    print(f"Selected model: {selected_name}")
    print(f"Saved pipeline: {artifact_path}")
    print(f"Actual-row sample prediction: {sample_prediction:,.0f}")


if __name__ == "__main__":
    main()
