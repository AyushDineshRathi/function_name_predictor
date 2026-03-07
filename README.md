# Function Name Prediction

Lightweight ML system that predicts a function name from metadata using:
- TF-IDF vectorization
- Logistic Regression classifier
- FastAPI for inference

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Full Pipeline

```bash
python run_pipeline.py
```

This generates:
- `models/function_model.pkl`
- `models/vectorizer.pkl`
- `reports/model_metrics.txt`
- `reports/model_size.txt`

## Run API

```bash
uvicorn src.api.app:app --reload
```

## API Usage

### Predict Endpoint

`POST /predict`

Request:

```json
{
  "metadata": "Adds two integers int a int b return int keywords add sum"
}
```

Response:

```json
{
  "predicted_function": "addNumbers"
}
```

Request:

```json
{
  "metadata": "Converts temperature from Celsius to Fahrenheit float celsius return float keywords convert temperature"
}
```

Response:

```json
{
  "predicted_function": "convertCelsiusToFahrenheit"
}
```

### Health Endpoint

`GET /health`

Response:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

## CLI Prediction

```bash
python predict_cli.py "Adds two integers int a int b return int keywords add sum"
```

Output:

```text
Predicted Function: addNumbers
```

## Inference Speed Benchmark

```bash
python src/inference/predict.py
```

This writes latency in milliseconds to:
- `reports/inference_speed.txt`
