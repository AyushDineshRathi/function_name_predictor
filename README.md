# Lightweight Function Name Prediction from Metadata

A compact machine learning system that predicts function names from metadata text.

Core capabilities:
- Structured metadata preprocessing
- TF-IDF features (`ngram_range=(1,3)`, `min_df=2`, English stopword removal)
- Primary model: `LinearSVC` (with `MultinomialNB` baseline comparison)
- Fast inference API with:
  - top-3 predictions
  - confidence scores
  - low-confidence rule-based function-name fallback
- Robust artifact handling (hash checks + sklearn version compatibility)

## Project Layout

```text
function_name_predictor/
|-- README.md
|-- .gitignore
`-- function-name-prediction/
    |-- run_pipeline.py
    |-- predict_cli.py
    |-- requirements.txt
    |-- data/
    |   |-- raw/
    |   `-- processed/
    |-- models/
    |-- reports/
    `-- src/
        |-- api/
        |-- data/
        |-- features/
        |-- inference/
        |-- models/
        `-- preprocessing/
```

## Setup

From repository root:

```bash
cd function-name-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Training Pipeline

```bash
cd function-name-prediction
python run_pipeline.py
```

Pipeline steps:
1. Generate raw synthetic dataset
2. Build structured metadata text
3. Remove duplicates
4. Stratified train/test split
5. Train `LinearSVC` and `MultinomialNB`
6. Select best model using policy
7. Save model/vectorizer atomically + manifest
8. Write metrics/size reports

Generated artifacts:
- `models/function_model.pkl`
- `models/vectorizer.pkl`
- `models/artifacts_manifest.json`
- `reports/model_metrics.txt`
- `reports/model_size.txt`

## Model Selection Policy

`LinearSVC` is preferred for sparse TF-IDF generalization.

`MultinomialNB` is trained as a baseline and overrides only when SVM is significantly worse:
- choose NB only if `NB_F1 - SVM_F1 > 0.05`

## Run API

```bash
cd function-name-prediction
uvicorn src.api.app:app --reload
```

Swagger docs:
- `http://127.0.0.1:8000/docs`

## API Usage

### `POST /predict`

Request:

```json
{
  "metadata": "Adds two integers int a int b return int keywords add sum"
}
```

Response shape:

```json
{
  "predicted_function": "addNumbers",
  "best_label": "addNumbers",
  "best_confidence": 0.2833,
  "threshold": 0.25,
  "top_predictions": [
    {"label": "addNumbers", "confidence": 0.2833},
    {"label": "calculateSum", "confidence": 0.0648},
    {"label": "multiplyNumbers", "confidence": 0.0376}
  ]
}
```

Low-confidence rule:
- If `best_confidence < threshold`, inference returns a lightweight generated name such as `startRunning` or `streamHeartRate`.

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## CLI Inference

```bash
cd function-name-prediction
python predict_cli.py "Adds two integers int a int b return int keywords add sum"
```

## Inference Benchmark

```bash
cd function-name-prediction
python src/inference/predict.py
```

Writes latency report to:
- `reports/inference_speed.txt`

## Notes

- If artifact compatibility/integrity checks fail, retrain:
  - `python run_pipeline.py`
- Avoid manually editing files under `models/`.
- `data/raw/improved_functions_1.csv` can be used for external-domain testing and future data augmentation.
