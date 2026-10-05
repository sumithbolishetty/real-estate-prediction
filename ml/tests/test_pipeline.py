from pathlib import Path
import json
import joblib
import pandas as pd
import pytest
from preprocessing import clean_dataset, to_model_frame, FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[2]


def test_dataset_cleaning_and_amenity_representation():
    frame = pd.read_csv(ROOT / "data/raw/sample_properties.csv")
    clean, report = clean_dataset(frame)
    assert frame.shape == (42, 9)
    assert report["source_rows"] == report["rows_after_cleaning"] == 42
    assert list(clean.columns) == FEATURE_COLUMNS + ["price"]
    assert len(clean) == report["rows_after_cleaning"]
    assert isinstance(clean.iloc[0]["amenities"], list)
    assert to_model_frame(clean).shape[1] == len(FEATURE_COLUMNS)


def test_saved_pipeline_predicts_single_row():
    model_path = ROOT / "artifacts/model/price_pipeline.joblib"
    assert model_path.is_file()
    model = joblib.load(model_path)
    frame = pd.read_csv(ROOT / "data/raw/sample_properties.csv")
    clean, _ = clean_dataset(frame)
    prediction = model.predict(to_model_frame(clean.iloc[:1]))
    assert prediction.shape == (1,)
    assert prediction[0] > 0


def test_cv_comparison_baseline_and_model_metadata():
    evaluation = json.loads((ROOT / "data/processed/evaluation.json").read_text())
    metadata_path = ROOT / "artifacts/model/model_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    preprocessing_metadata = json.loads((ROOT / "artifacts/preprocessing/metadata.json").read_text())
    assert evaluation["folds"] == 5
    assert "Mean Price Baseline" in evaluation["models"]
    assert metadata["training_rows"] == 42
    assert metadata["feature_names"] == FEATURE_COLUMNS
    assert metadata["target_name"] == "price"
    assert preprocessing_metadata["model_name"] == evaluation["selected_model"]
    assert len(evaluation["models"][evaluation["selected_model"]]["folds"]) == 5
    selected_mae = evaluation["models"][evaluation["selected_model"]]["summary"]["MAE"]["mean"]
    baseline_mae = evaluation["models"]["Mean Price Baseline"]["summary"]["MAE"]["mean"]
    assert selected_mae < baseline_mae
    assert metadata["feature_importance"]["by_feature_group"]
