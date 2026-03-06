from pathlib import Path
import re
import time
import warnings
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "function_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
INFERENCE_SPEED_REPORT_PATH = REPORTS_DIR / "inference_speed.txt"

# Global variables to cache the model and vectorizer
_model = None
_vectorizer = None

def resources_loaded() -> bool:
    return _model is not None and _vectorizer is not None

def _load_pickle_with_version_check(file_path: Path, artifact_name: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = joblib.load(file_path)

    version_warnings = [w for w in caught if w.category.__name__ == "InconsistentVersionWarning"]
    if version_warnings:
        raise RuntimeError(
            f"{artifact_name} is incompatible with current scikit-learn version. "
            "Please run: python run_pipeline.py"
        )
    return loaded

def normalize_text(text: str) -> str:
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def load_resources(model_path=None, vectorizer_path=None):
    """
    Loads the trained model and vectorizer from disk.
    Cached after the first call to avoid reloading on subsequent predictions.
    """
    global _model, _vectorizer
    
    if model_path is None:
        model_path = MODEL_PATH
    if vectorizer_path is None:
        vectorizer_path = VECTORIZER_PATH

    model_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)
        
    if _model is None:
        if not model_file.exists():
            raise FileNotFoundError("Models not found. Please run: python run_pipeline.py")
        _model = _load_pickle_with_version_check(model_file, "Model file")
        
    if _vectorizer is None:
        if not vectorizer_file.exists():
            raise FileNotFoundError("Models not found. Please run: python run_pipeline.py")
        _vectorizer = _load_pickle_with_version_check(vectorizer_file, "Vectorizer file")

def predict_function(metadata_text: str) -> str:
    """
    Transforms the input metadata text and returns the predicted function name.
    
    Args:
        metadata_text (str): The combined and processed metadata string.
        
    Returns:
        str: The predicted function name.
    """
    # Ensure models are loaded
    load_resources()

    normalized_text = normalize_text(metadata_text)
    
    # 1. Transform input metadata text using vectorizer
    # The vectorizer expects an iterable of strings
    transformed_text = _vectorizer.transform([normalized_text])
    
    # 2. Run model prediction
    prediction = _model.predict(transformed_text)
    
    # 3. Return predicted function name
    return prediction[0]

def benchmark_inference(sample_text: str) -> tuple:
    # Warm-up call to avoid counting one-time model/vectorizer load time.
    _ = predict_function(sample_text)

    start = time.perf_counter()
    prediction = predict_function(sample_text)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = (
        "Inference Speed Report\n"
        "======================\n"
        f"Input: {sample_text}\n"
        f"Prediction: {prediction}\n"
        f"Latency: {elapsed_ms:.4f} ms\n"
    )
    INFERENCE_SPEED_REPORT_PATH.write_text(report_content, encoding="utf-8")
    return prediction, elapsed_ms

if __name__ == "__main__":
    import sys
    
    # Ensure standard paths resolve correctly inside the script too
    sys.path.append(str(PROJECT_ROOT))
    
    print("\n--- Function Name Predictor (Inference Demonstration) ---")
    
    # Example input used for lightweight benchmark
    example_input = "Adds two integers int a int b return int keywords add sum"
    print(f"\nInput Metadata: \"{example_input}\"")
    
    try:
        predicted_name, latency_ms = benchmark_inference(example_input)
        print("Expected output example:")
        print(f"Predicted Function: {predicted_name}\n")
        print(f"Inference latency: {latency_ms:.4f} ms")
        print(f"Speed report saved to: {INFERENCE_SPEED_REPORT_PATH}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
