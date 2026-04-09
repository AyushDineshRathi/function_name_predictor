# Lightweight Function Name Prediction from Metadata

This project predicts function names from metadata using a lightweight machine learning pipeline and a FastAPI service.

## Project Overview

The system is designed for fast, practical inference with compact artifacts:
- Metadata preprocessing into a structured text format
- TF-IDF feature extraction (`ngram_range=(1,3)`, `min_df=2`, English stopword removal)
- Primary classifier: `LinearSVC`
- Baseline comparison: `MultinomialNB`
- Top-3 predictions with confidence scores
- Rule-based fallback name generation for low-confidence cases

## Repository Structure

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

From the repository root:

```bash
cd function-name-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Training Pipeline

Run the full pipeline:

```bash
cd function-name-prediction
python run_pipeline.py
```

What this does:
1. Generates raw synthetic dataset
2. Preprocesses metadata into structured training text
3. Removes duplicates and performs stratified train/test split
4. Trains TF-IDF vectorizer
5. Trains `LinearSVC` and `MultinomialNB`
6. Selects best model by policy and saves artifacts
7. Writes evaluation and size reports

Generated artifacts:
- `models/function_model.pkl`
- `models/vectorizer.pkl`
- `models/artifacts_manifest.json`
- `reports/model_metrics.txt`
- `reports/model_size.txt`

## Model Selection Policy

`LinearSVC` is prioritized for better generalization on sparse TF-IDF features.

`MultinomialNB` is selected only when SVM is significantly worse:
- choose NB if `NB_F1 - SVM_F1 > 0.05`

## Run API

```bash
cd function-name-prediction
python src/api/app.py
```

Or with Uvicorn:

```bash
cd function-name-prediction
uvicorn src.api.app:app --reload
```

API docs:
- `http://127.0.0.1:8000/docs`

## API Endpoints

### `POST /predict`

Request:

```json
{
  "metadata": "Adds two integers int a int b return int keywords add sum"
}
```

Response format:

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

### `GET /health`

Response:

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

Output file:
- `reports/inference_speed.txt`
