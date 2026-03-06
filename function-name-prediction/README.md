# Function Name Prediction

A lightweight machine learning system designed to predict an appropriate function name from structured function metadata. This project prioritizes small model size, fast inference, and easy deployment, making it suitable for environments with limited computational resources (such as Android devices).

## Project Structure Overview

- `data/`
  - `raw/`: Raw metadata for training.
  - `processed/`: Processed data ready to be consumed by the training models.
- `src/`: Modular source code for the ML pipeline.
  - `data/`: Scripts for local data loading and splitting.
  - `preprocessing/`: Code to clean text metadata and combine fields.
  - `features/`: Vectorization and feature extraction logic (e.g., TF-IDF).
  - `models/`: Implementations for lightweight classifiers (Logistic Regression / Naive Bayes).
  - `evaluation/`: Scripts and metrics to evaluate model performance to ensure accuracy.
  - `inference/`: Prediction logic for making inferences on new data.
  - `api/`: Placeholder for future deployment modules.
- `models/`: Directory holding compiled and saved models (`.pkl` or `.joblib` format).
- `notebooks/`: Jupyter notebooks for exploratory data analysis, prototyping, and validation.

## Environment Setup

The dependencies for this project are minimal by design, relying primarily on `pandas`, `scikit-learn`, `numpy`, and `joblib`.

1. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Running the Pipeline

Currently, the primary entry point is being structured inside `run_pipeline.py`, which will eventually orchestrate reading raw data, performing validations, applying preprocessing, fitting the TF-IDF vectorizer, training the targeted model, evaluating it, and serializing it into the `models/` directory.

To run the pipeline outline (when complete):
```bash
python run_pipeline.py
```
