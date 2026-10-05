# Real Estate Price Prediction System

## 1. Project overview

A local demonstration app estimates a property's price from a small set of property details using a scikit-learn regression model, a FastAPI backend, and a Next.js frontend. An estimate is not a guaranteed market price or official valuation. The interface explains that actual prices vary with factors not represented in this dataset.

## 2. Features

- User selects location, area in sq ft, bedroom/bathroom counts, property type, parking, property age in years, and optional amenities.
- API-backed dropdown choices come from the dataset categories saved with the model.
- Request validation, clear errors, loading feedback, and Indian currency display.
- The app has no database, account system, Docker dependency, or separate ML server.

## 3. Architecture

`Browser (localhost:3100) → FastAPI (127.0.0.1:8100) → saved sklearn preprocessing/model pipeline → estimated price`.
The browser reads `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8100`) for `/api/options` and `/api/predict`. The API allows the frontend origins `http://localhost:3100` and `http://127.0.0.1:3100`.

## 4. Project structure

```text
frontend/                     Next.js UI, styles, and frontend instructions
backend/app/                  FastAPI routes, request schema, artifact loading
backend/tests/                API validation and model loading tests
ml/                           preprocessing, training, evaluation, prediction
ml/tests/                     data, preprocessing, artifact, prediction tests
data/raw/                     synthetic development CSV
data/processed/               generated dataset and evaluation reports
artifacts/model/              generated trained pipeline
artifacts/preprocessing/      generated model/category metadata
```

## 5. Dataset

The current dataset is `data/raw/sample_properties.csv`, with 42 rows and 9 columns. It is the existing project-created demo dataset, not independently verified real-market data. No larger compatible dataset exists under `data/` or `ml/`; it has not been expanded, duplicated, or supplemented with fabricated records. Target: `price` in rupees. The eight input feature groups are `location`, `area`, `bedrooms`, `bathrooms`, `property_type`, `parking`, `property_age`, and `amenities`. Training records the actual row/column counts, data types, missing values, ranges, observed categories, cleaning counts, and outlier analysis in `data/processed/dataset_report.json`.

The dataset's location, property type, and parking categories are inspected from its values and exposed by `/api/options`; no extra categories are fabricated. Amenity text is split on pipe/comma/semicolon delimiters, canonicalized against an allowlist, and encoded as multi-hot features. Category text is trimmed, case-folded, and whitespace-normalized. Area is square feet throughout. Property age is years. Parking is categorical yes/no.

## 6. ML pipeline and cleaning

The shared `ml/preprocessing.py` definition validates the eight input groups and the `price` target. It reports and removes exact duplicate rows, rows missing required values, blank required categories, nonnumeric/non-finite numeric values, area at or below zero, bedrooms below one, bathrooms at or below zero, negative age, and nonpositive price. No rows from this dataset required removal. Three high-priced records are flagged by the IQR screen; their premium sizes/types do not show a clear data-entry error, so they are retained. The data is demo material, so real-world legitimacy cannot be independently established. Category case and whitespace are normalized. Pipe-delimited amenity lists are canonicalized and multi-hot encoded within the saved pipeline, which is reused at inference. There is no silent unit conversion.

## 7. Model and evaluation

The training script compares a training-fold-mean baseline with Random Forest, Extra Trees, and Hist Gradient Boosting using the same shuffled five-fold cross-validation splits (`random_state=42`). It selects the regressor with the lowest mean fold MAE, then refits its saved preprocessing/model pipeline on all 42 valid rows. Reports include mean and standard deviation across folds for MAE (average absolute price error), RMSE (root mean squared error), and R². The current measured results are in `data/processed/evaluation.json`; run `cd ml && source .venv/bin/activate && python train.py` to recalculate them. The current dataset is relatively small, so evaluation results should be interpreted cautiously. Fold results and feature-importance rankings are uncertain and do not establish causation or production accuracy.

Current five-fold results (mean ± standard deviation across folds; rupee error metrics):

| Candidate | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean price baseline | ₹8,116,410 ± ₹2,180,268 | ₹10,783,479 ± ₹4,319,393 | -0.213 ± 0.224 |
| Random Forest Regression | ₹4,431,423 ± ₹2,608,040 | ₹6,427,405 ± ₹3,973,787 | 0.572 ± 0.337 |
| Extra Trees Regression | ₹3,768,013 ± ₹1,698,884 | ₹6,165,657 ± ₹3,186,914 | 0.508 ± 0.459 |
| Hist Gradient Boosting Regression | ₹4,645,287 ± ₹2,234,264 | ₹7,016,507 ± ₹3,526,971 | 0.445 ± 0.453 |

Extra Trees is selected because its measured mean CV MAE is lowest among the regression models and lower than the mean baseline. Random Forest has a higher average R², so selection depends on the stated MAE criterion. The fold variation is substantial, especially for R².

The selected tree model's grouped and encoded feature importances are included in the evaluation and model metadata reports. On the all-data Extra Trees fit, group importances are amenities 38.9%, location 18.7%, area 17.8%, bathrooms 17.7%, bedrooms 3.1%, property type 2.3%, property age 1.1%, and parking 0.4%. These small-data rankings describe model reliance only, not causal effects, and can change substantially with more legitimate observations.

## ML Methodology

1. **Data Collection:** Use the existing 42-row `sample_properties.csv`; no compatible larger project dataset was present.
2. **Data Cleaning:** Inspect schema, missing values, duplicates, invalid numeric values, categories, and amenity tokens. Remove only invalid rows; normalize categorical spelling/capitalization and retain valid IQR outliers.
3. **Exploratory Data Analysis:** Record numerical ranges, category counts, price distribution, IQR flags, feature variation, and descriptive numeric correlations in the dataset report.
4. **Feature Engineering:** Split the pipe-delimited amenity list into the eight supported amenity flags; keep area in square feet and age in years.
5. **Preprocessing:** Use one scikit-learn pipeline for numeric pass-through, one-hot categorical encoding, and multi-hot amenity encoding. Training and inference share the same pipeline.
6. **Cross-Validation:** Use shuffled five-fold K-fold validation because a single holdout split is unstable for 42 rows; report each fold and mean ± standard deviation.
7. **Model Training:** Evaluate Random Forest, Extra Trees, and Hist Gradient Boosting; also evaluate a mean-price baseline.
8. **Model Comparison:** Select the regression model with the lowest mean cross-validation MAE, compared on identical folds to the baseline.
9. **Evaluation:** Report MAE, RMSE, and R² with fold variation. Interpret all metrics cautiously given the sample size.
10. **Model Deployment:** Save the fitted pipeline under `artifacts/model/`, metadata under `artifacts/model/model_metadata.json`, and category/preprocessing metadata under `artifacts/preprocessing/` for FastAPI.
11. **Prediction:** Validate the request, normalize supported categorical values, run the saved pipeline, and display the estimated price in Indian currency.

**Limitation:** The available dataset contains only 42 records, which limits the generalizability of the trained model.

## 8. Backend API

- `GET /api/health` → `{"status":"ok","model_loaded":true}`
- `GET /api/options` → supported `locations`, `property_types`, `parking_options`, `amenities`
- `POST /api/predict` → validated estimate and Indian-formatted price
- Interactive docs: `http://127.0.0.1:8100/docs`

Example request:

```json
{
  "location": "pune",
  "area": 1200,
  "bedrooms": 3,
  "bathrooms": 2,
  "property_type": "apartment",
  "parking": "yes",
  "property_age": 5,
  "amenities": ["Gym", "Lift"]
}
```

Example response shape: `{"estimated_price": 8500000, "formatted_price": "₹85,00,000"}`. Values shown here illustrate the response format, not a model result. Malformed requests and unsupported categories receive a 4xx response; unavailable artifacts result in a clear non-healthy state/API error. Extra request fields are rejected.

## 9. Frontend

The form has client-side checks and controlled amenity selections, displays progress, and formats prices with Indian digit grouping. It shows a result disclaimer that estimated prices vary with factors absent from the data. Backend options populate form dropdowns, keeping categories aligned.

## 10. Installation

Use separate Python environments. Commands below start from the project root unless a subsection explicitly says to remain in a directory.

### ML dependencies and training

```sh
cd ml
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python evaluate.py
```

The training step generates the pipeline at `artifacts/model/price_pipeline.joblib`, model/evaluation/feature-importance metadata at `artifacts/model/model_metadata.json`, supported category and preprocessing metadata at `artifacts/preprocessing/metadata.json`, and inspection/evaluation reports in `data/processed/`. It reloads the final artifact and validates a positive numeric prediction for an actual dataset row before exiting successfully. Restart the backend after training so its cached model loader uses the new artifact.

### Backend installation and start

From project root:

```sh
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8100
```

The same uvicorn command is used when already inside `backend/`; do not change directories again. Backend: `http://127.0.0.1:8100`.

### Frontend installation and start

From project root, in a second terminal:

```sh
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Frontend: `http://localhost:3100`. The npm dev script explicitly starts Next.js on port 3100. Frontend runs on port 3100 and backend runs on port 8100 to prevent conflicts with previous projects.

## 11. Testing and verification

After training and installing the backend dependencies, from `backend/` run `python -m pip install pytest httpx && python -m pytest`. From `ml/`, install pytest with `python -m pip install pytest` and run `python -m pytest tests`. Tests cover health, artifact loading, options, successful inference, input/category rejection, dataset cleaning, preprocessing, and artifact predictions. To check CORS, send an OPTIONS preflight to the API with `Origin: http://localhost:3100`; the response should include that allowed origin. To check the complete browser flow, keep the backend and frontend running, load the frontend, and submit a supported property.

## 12. Port configuration

Only the frontend port 3100 and backend port 8100 are configured. The frontend API base URL is configurable with `NEXT_PUBLIC_API_URL`; its local example points to `http://127.0.0.1:8100`. The backend command binds specifically to `127.0.0.1:8100`. Port conflicts should be resolved by identifying and stopping the process occupying the configured port; the app does not silently switch ports. On macOS, inspect a port with `lsof -nP -iTCP:<port> -sTCP:LISTEN`, identify the process, then stop that process deliberately.

## 13. Troubleshooting

- **`python: command not found`:** Use `python3` to create the virtual environment; after activating it, use `python -m pip` and `python -m uvicorn`.
- **`cd: no such file or directory`:** Start at the project root for `cd ml`, `cd backend`, or `cd frontend`; if already in a component directory, run its commands there without another `cd`.
- **Port already in use:** Inspect the configured port with the macOS `lsof` command above, then stop the identified process. The application keeps its configured ports.
- **Frontend showing an old project:** Open `http://localhost:3100`, confirm the frontend terminal is running in this project's `frontend/`, and check its `npm run dev` output.
- **Backend unavailable / CORS error:** Confirm the backend is running at `127.0.0.1:8100`; open the frontend at `localhost:3100` so its origin matches the explicit CORS allowlist.
- **Model file not found:** Train from `ml/` with its environment activated. The backend requires the generated model and metadata before startup.
- **Unknown location / property type:** Select one of the options returned by `GET /api/options`. The API intentionally rejects categories absent from training data.
- **Prediction request failed:** Check the API health route, required form values, and browser network request; the app displays a friendly error instead of a traceback.
- **`npm` command not found:** Install Node.js with npm, reopen the terminal, and confirm `node --version` and `npm --version` are available.

## 14. Limitations

This project uses a small synthetic sample. It does not provide a credible valuation, capture local market conditions or temporal changes, or support locations and categories absent from the sample. Replace the demo CSV with a legally usable real dataset, inspect its schema and quality, and retrain before using predictions for anything beyond demonstration.

## 15. Isolation

This standalone project does not import another project's code, dataset, model, environment, API URL, or process. It has no Docker files or dependency. Configured application ports: frontend **3100**, backend **8100**.
