"""Print the cross-validation report produced by train.py."""
import json
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "data/processed/evaluation.json"
if not path.exists():
    raise SystemExit("Evaluation report missing. From ml/, run python train.py first.")
report = json.loads(path.read_text())
print("Five-fold cross-validation (mean ± fold standard deviation)")
print("-----------------------------------------------------------")
for model_name, result in report["models"].items():
    metrics = result["summary"]
    print(
        f"{model_name}: "
        f"MAE {metrics['MAE']['mean']:,.0f} ± {metrics['MAE']['std']:,.0f}; "
        f"RMSE {metrics['RMSE']['mean']:,.0f} ± {metrics['RMSE']['std']:,.0f}; "
        f"R² {metrics['R2']['mean']:.3f} ± {metrics['R2']['std']:.3f}"
    )
print(f"Selected model: {report['selected_model']}")
print("Feature importance by group:")
for feature, value in report["feature_importance"]["by_feature_group"].items():
    print(f"  {feature}: {value:.3f}")
print(report["uncertainty_note"])
