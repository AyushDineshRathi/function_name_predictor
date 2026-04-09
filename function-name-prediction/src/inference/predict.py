from pathlib import Path
import time
import warnings
import joblib
import json
import hashlib
import numpy as np
from sklearn import __version__ as sklearn_version
from sklearn.utils.validation import check_is_fitted
from src.preprocessing.text_normalizer import build_structured_metadata_from_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "function_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"
MANIFEST_PATH = PROJECT_ROOT / "models" / "artifacts_manifest.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
INFERENCE_SPEED_REPORT_PATH = REPORTS_DIR / "inference_speed.txt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25

ACTION_MAP = {
    "begin": "start",
    "begins": "start",
    "start": "start",
    "starts": "start",
    "end": "stop",
    "ends": "stop",
    "stop": "stop",
    "stops": "stop",
    "enable": "enable",
    "enables": "enable",
    "disable": "disable",
    "disables": "disable",
    "retrieve": "get",
    "retrieves": "get",
    "stream": "stream",
    "streams": "stream",
}

OBJECT_PHRASE_MAP = [
    ("heart rate", "HeartRate"),
    ("running workout session", "Running"),
    ("running workout", "Running"),
    ("running session", "Running"),
    ("yoga workout session", "Yoga"),
    ("yoga workout", "Yoga"),
    ("temperature", "Temperature"),
    ("music", "Music"),
    ("running", "Running"),
    ("yoga", "Yoga"),
]

# Cached artifacts loaded once per process.
_model = None
_vectorizer = None


def resources_loaded() -> bool:
    return _model is not None and _vectorizer is not None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise RuntimeError("Artifact manifest missing. Please run: python run_pipeline.py")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_keys = {"sklearn_version", "model", "vectorizer"}
    if not required_keys.issubset(data):
        raise RuntimeError("Artifact manifest is corrupted. Please run: python run_pipeline.py")
    return data


def _validate_manifest_consistency(manifest: dict, model_file: Path, vectorizer_file: Path) -> None:
    if manifest.get("sklearn_version") != sklearn_version:
        raise RuntimeError(
            "Artifact sklearn version mismatch. "
            f"Artifacts: {manifest.get('sklearn_version')}, Runtime: {sklearn_version}. "
            "Please run: python run_pipeline.py"
        )

    expected_model_hash = manifest["model"].get("sha256")
    expected_vec_hash = manifest["vectorizer"].get("sha256")
    if not expected_model_hash or not expected_vec_hash:
        raise RuntimeError("Artifact manifest hashes are missing/corrupted. Please run: python run_pipeline.py")

    model_hash = _sha256_file(model_file)
    vec_hash = _sha256_file(vectorizer_file)
    if model_hash != expected_model_hash or vec_hash != expected_vec_hash:
        raise RuntimeError("Artifact integrity check failed. Please run: python run_pipeline.py")


def _validate_loaded_objects(model, vectorizer) -> None:
    check_is_fitted(vectorizer, ["vocabulary_", "idf_"])
    check_is_fitted(model)

    if not hasattr(model, "predict") or not hasattr(model, "classes_"):
        raise RuntimeError("Corrupted model artifact loaded. Please run: python run_pipeline.py")
    if not hasattr(vectorizer, "transform") or not hasattr(vectorizer, "idf_"):
        raise RuntimeError("Corrupted vectorizer artifact loaded. Please run: python run_pipeline.py")

    probe = vectorizer.transform(["artifact integrity probe"])
    model_dim = getattr(model, "n_features_in_", probe.shape[1])
    if probe.shape[1] != model_dim:
        raise RuntimeError("Model/vectorizer feature mismatch. Please run: python run_pipeline.py")


def _load_pickle_with_version_check(file_path: Path, artifact_name: str):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = joblib.load(file_path)

    if caught:
        warning_types = ", ".join(sorted({w.category.__name__ for w in caught}))
        raise RuntimeError(
            f"{artifact_name} emitted load warnings ({warning_types}); "
            "artifact may be incompatible or corrupted. "
            "Please run: python run_pipeline.py"
        )
    return loaded


def normalize_text(text: str) -> str:
    return build_structured_metadata_from_text(text)


def _to_pascal_case(text: str) -> str:
    tokens = [t for t in text.split() if t]
    return "".join(token[:1].upper() + token[1:] for token in tokens)


def _extract_action(raw_text: str) -> str:
    for token in raw_text.lower().split():
        mapped = ACTION_MAP.get(token)
        if mapped:
            return mapped
    return "identify"


def _extract_object(raw_text: str) -> str:
    text = raw_text.lower()
    for phrase, normalized in OBJECT_PHRASE_MAP:
        if phrase in text:
            return normalized

    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in raw_text)
    tokens = [t for t in cleaned.split() if t]
    blocked = set(ACTION_MAP.keys()) | {
        "a",
        "an",
        "the",
        "to",
        "from",
        "of",
        "for",
        "and",
        "or",
        "with",
        "recording",
        "real",
        "time",
        "data",
        "workout",
        "session",
    }
    candidates = [t for t in tokens if t not in blocked and len(t) > 2]
    if not candidates:
        return "Result"
    return _to_pascal_case(candidates[0])


def generate_rule_based_function_name(metadata_text: str) -> str:
    raw_text = str(metadata_text or "")
    action = _extract_action(raw_text)
    obj = _extract_object(raw_text)
    return f"{action}{obj}"


def load_resources(model_path=None, vectorizer_path=None):
    global _model, _vectorizer

    model_file = Path(model_path) if model_path is not None else MODEL_PATH
    vectorizer_file = Path(vectorizer_path) if vectorizer_path is not None else VECTORIZER_PATH

    if not model_file.exists() or not vectorizer_file.exists():
        raise FileNotFoundError("Models not found. Please run: python run_pipeline.py")

    if _model is None or _vectorizer is None:
        manifest = _load_manifest(MANIFEST_PATH)
        _validate_manifest_consistency(manifest, model_file, vectorizer_file)

        loaded_model = _load_pickle_with_version_check(model_file, "Model file")
        loaded_vectorizer = _load_pickle_with_version_check(vectorizer_file, "Vectorizer file")
        _validate_loaded_objects(loaded_model, loaded_vectorizer)

        _model = loaded_model
        _vectorizer = loaded_vectorizer


def predict_with_confidence(metadata_text: str, top_k: int = 3, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> dict:
    load_resources()
    normalized_text = normalize_text(metadata_text)
    transformed_text = _vectorizer.transform([normalized_text])

    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(transformed_text)[0]
    elif hasattr(_model, "decision_function"):
        scores = np.asarray(_model.decision_function(transformed_text))
        if scores.ndim == 1:
            pos = scores[0]
            scores = np.array([-pos, pos], dtype=float)
        else:
            scores = scores[0].astype(float)

        max_score = np.max(scores)
        exp_scores = np.exp(scores - max_score)
        denom = np.sum(exp_scores)
        proba = exp_scores / denom if denom > 0 else np.ones_like(exp_scores) / len(exp_scores)
    else:
        raise RuntimeError("Loaded model does not support confidence scoring (missing predict_proba/decision_function).")

    classes = np.asarray(_model.classes_)
    k = max(1, min(top_k, len(proba)))
    top_idx_unsorted = np.argpartition(proba, -k)[-k:]
    top_idx = top_idx_unsorted[np.argsort(proba[top_idx_unsorted])[::-1]]

    top_predictions = [{"label": str(classes[i]), "confidence": float(proba[i])} for i in top_idx]
    best_label = top_predictions[0]["label"]
    best_confidence = top_predictions[0]["confidence"]

    if best_confidence >= threshold:
        predicted_function = best_label
    else:
        predicted_function = generate_rule_based_function_name(metadata_text)

    return {
        "predicted_function": predicted_function,
        "best_label": best_label,
        "best_confidence": best_confidence,
        "threshold": float(threshold),
        "top_predictions": top_predictions,
    }


def predict_function(metadata_text: str) -> str:
    return predict_with_confidence(metadata_text, top_k=3, threshold=DEFAULT_CONFIDENCE_THRESHOLD)["predicted_function"]


def benchmark_inference(sample_text: str):
    _ = predict_with_confidence(sample_text)
    start = time.perf_counter()
    result = predict_with_confidence(sample_text)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_content = (
        "Inference Speed Report\n"
        "======================\n"
        f"Input: {sample_text}\n"
        f"Prediction: {result['predicted_function']}\n"
        f"Best Label: {result['best_label']}\n"
        f"Best Confidence: {result['best_confidence']:.4f}\n"
        f"Latency: {elapsed_ms:.4f} ms\n"
    )
    INFERENCE_SPEED_REPORT_PATH.write_text(report_content, encoding="utf-8")
    return result["predicted_function"], elapsed_ms


if __name__ == "__main__":
    print("\n--- Function Name Predictor (Inference Demonstration) ---")
    example_input = "Adds two integers int a int b return int keywords add sum"
    print(f"\nInput Metadata: \"{example_input}\"")
    try:
        predicted_name, latency_ms = benchmark_inference(example_input)
        print(f"Predicted Function: {predicted_name}")
        print(f"Inference latency: {latency_ms:.4f} ms")
        print(f"Speed report saved to: {INFERENCE_SPEED_REPORT_PATH}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
