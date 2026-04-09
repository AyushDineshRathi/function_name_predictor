import sys
from pathlib import Path
import pandas as pd
import joblib
import json
import hashlib
from datetime import datetime, timezone
from uuid import uuid4
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn import __version__ as sklearn_version
from sklearn.utils.validation import check_is_fitted

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "function_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"
MANIFEST_PATH = PROJECT_ROOT / "models" / "artifacts_manifest.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_REPORT_PATH = REPORTS_DIR / "model_metrics.txt"
SIZE_REPORT_PATH = REPORTS_DIR / "model_size.txt"

# Add the project root to sys.path so we can import src modules
sys.path.append(str(PROJECT_ROOT))

from src.features.vectorizer import train_vectorizer, transform_text, get_vectorizer


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_fitted_artifacts(model, vectorizer) -> None:
    check_is_fitted(vectorizer, ["vocabulary_", "idf_"])
    check_is_fitted(model)

    if not hasattr(model, "predict"):
        raise RuntimeError("Corrupted model artifact: missing predict().")
    if not hasattr(model, "classes_"):
        raise RuntimeError("Corrupted model artifact: missing classes_.")
    if not hasattr(vectorizer, "transform"):
        raise RuntimeError("Corrupted vectorizer artifact: missing transform().")

    probe = vectorizer.transform(["artifact integrity probe"])
    model_dim = getattr(model, "n_features_in_", probe.shape[1])
    if probe.shape[1] != model_dim:
        raise RuntimeError(
            f"Incompatible artifacts: vectorizer output dim {probe.shape[1]} != model dim {model_dim}."
        )


def _atomic_save_artifacts(model, vectorizer) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    training_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    model_tmp = MODEL_PATH.with_suffix(f".tmp.{training_id}.pkl")
    vectorizer_tmp = VECTORIZER_PATH.with_suffix(f".tmp.{training_id}.pkl")
    manifest_tmp = MANIFEST_PATH.with_suffix(f".tmp.{training_id}.json")

    try:
        joblib.dump(model, model_tmp)
        joblib.dump(vectorizer, vectorizer_tmp)

        manifest = {
            "training_id": training_id,
            "created_at_utc": created_at,
            "sklearn_version": sklearn_version,
            "model": {
                "file": MODEL_PATH.name,
                "sha256": _sha256_file(model_tmp),
                "class_name": type(model).__name__,
            },
            "vectorizer": {
                "file": VECTORIZER_PATH.name,
                "sha256": _sha256_file(vectorizer_tmp),
                "class_name": type(vectorizer).__name__,
            },
        }

        manifest_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Swap data files first, then publish manifest last.
        # Inference only trusts artifacts that match the latest manifest.
        model_tmp.replace(MODEL_PATH)
        vectorizer_tmp.replace(VECTORIZER_PATH)
        manifest_tmp.replace(MANIFEST_PATH)
    finally:
        for tmp_file in (model_tmp, vectorizer_tmp, manifest_tmp):
            if tmp_file.exists():
                tmp_file.unlink()

def load_data(path=PROCESSED_DATA_PATH):
    """
    Loads the processed dataset from the specified path.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
    print(f"Loading processed dataset from {data_path}...")
    return pd.read_csv(data_path)

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluates the model and prints accuracy, precision, recall, f1-score, and confusion matrix.
    Returns the f1-score to help in deciding the best model.
    """
    print(f"\n--- Evaluation for {model_name} ---")
    
    # Using weighted average to handle multi-class classification
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }

def _format_metrics_block(metrics: dict) -> str:
    return (
        f"--- {metrics['model']} ---\n"
        f"Accuracy:  {metrics['accuracy']:.4f}\n"
        f"Precision: {metrics['precision']:.4f}\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"F1-score:  {metrics['f1']:.4f}\n"
        f"Confusion Matrix:\n{metrics['confusion_matrix']}\n"
    )

def main():
    # 1. Load processed dataset
    df = load_data()
    
    # Drop rows with missing values if any
    df = df.dropna(subset=['combined_metadata', 'function_name'])

    # Remove duplicates to improve dataset quality and prevent leakage.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["combined_metadata", "function_name"]).reset_index(drop=True)
    removed_dupes = before_dedup - len(df)
    print(f"\nRemoved duplicate rows: {removed_dupes}")
    print(f"Number of unique samples: {len(df)}")
    
    X_text = df['combined_metadata'].astype(str)
    y = df['function_name']

    class_distribution = y.value_counts().sort_index()
    print("\nClass distribution (unique samples):")
    print(class_distribution.to_string())
    
    # 2. Split data into train/test (80/20)
    print("\nSplitting data into train/test (80/20, stratified)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Explicit leakage guard: exact text overlap must be zero after dedup + split.
    overlap = set(X_train_text) & set(X_test_text)
    if overlap:
        raise RuntimeError(
            f"Train-test leakage detected: {len(overlap)} overlapping text samples between splits."
        )

    # 3. Train TF-IDF vectorizer on train split only, then transform test split
    X_train = train_vectorizer(X_train_text)
    X_test = transform_text(X_test_text)

    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # 4. Train Models
    print("\nTraining models...")
    
    # Primary model: Linear SVM (better margin-based generalization for sparse TF-IDF)
    svm_model = LinearSVC(class_weight="balanced", random_state=42)
    svm_model.fit(X_train, y_train)
    svm_preds = svm_model.predict(X_test)
    svm_metrics = evaluate_model(y_test, svm_preds, "Linear SVM")
    
    # Baseline comparison model: Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    nb_metrics = evaluate_model(y_test, nb_preds, "Multinomial Naive Bayes")
    
    # 5. Select Best Model
    print("\n==================================")
    print("Model Selection:")

    # Policy: prefer Linear SVM unless it is significantly worse than NB (>5% absolute F1 gap).
    f1_gap = nb_metrics["f1"] - svm_metrics["f1"]
    significant_worse = f1_gap > 0.05

    if significant_worse:
        best_name, best_model, best_metrics = (
            "Multinomial Naive Bayes",
            nb_model,
            nb_metrics,
        )
        print(
            "Linear SVM was not selected because its F1 is more than 5% lower than NB "
            f"(SVM={svm_metrics['f1']:.4f}, NB={nb_metrics['f1']:.4f}, gap={f1_gap:.4f})."
        )
    else:
        best_name, best_model, best_metrics = (
            "Linear SVM",
            svm_model,
            svm_metrics,
        )
        print(
            "Linear SVM selected by policy: margin-based model for sparse TF-IDF "
            "features with generally stronger generalization; NB kept only as baseline. "
            f"(SVM F1={svm_metrics['f1']:.4f}, NB F1={nb_metrics['f1']:.4f}, gap={f1_gap:.4f})"
        )

    print(f"Best model selected: {best_name}")
    print("Selected model metrics:")
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"F1-score:  {best_metrics['f1']:.4f}")
    print("==================================")
    
    # 6. Save consistent model + vectorizer artifacts atomically
    print(f"\nSaving consistent artifacts to {MODEL_PATH.parent}...")
    fitted_vectorizer = get_vectorizer()
    _validate_fitted_artifacts(best_model, fitted_vectorizer)
    _atomic_save_artifacts(best_model, fitted_vectorizer)

    # Verify selected model type matches what was saved.
    saved_model = joblib.load(MODEL_PATH)
    if type(saved_model).__name__ != type(best_model).__name__:
        raise RuntimeError(
            "Saved model type does not match selected model type. "
            f"Selected={type(best_model).__name__}, Saved={type(saved_model).__name__}"
        )
    print("Model + vectorizer saved successfully. Training pipeline complete.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_report = (
        "Model Evaluation Report\n"
        "=======================\n\n"
        f"{_format_metrics_block(svm_metrics)}\n"
        f"{_format_metrics_block(nb_metrics)}\n"
        f"Best model: {best_name}\n"
        f"Best model metrics: Accuracy={best_metrics['accuracy']:.4f}, "
        f"Precision={best_metrics['precision']:.4f}, "
        f"Recall={best_metrics['recall']:.4f}, F1={best_metrics['f1']:.4f}\n"
    )
    METRICS_REPORT_PATH.write_text(metrics_report, encoding="utf-8")
    print(f"Metrics report saved to {METRICS_REPORT_PATH}")

    model_size_kb = MODEL_PATH.stat().st_size / 1024
    vectorizer_size_kb = VECTORIZER_PATH.stat().st_size / 1024
    print(f"Model size (function_model.pkl): {model_size_kb:.2f} KB")
    print(f"Vectorizer size (vectorizer.pkl): {vectorizer_size_kb:.2f} KB")

    size_report = (
        "Model Size Report\n"
        "=================\n"
        f"function_model.pkl : {model_size_kb:.2f} KB\n"
        f"vectorizer.pkl : {vectorizer_size_kb:.2f} KB\n"
    )
    SIZE_REPORT_PATH.write_text(size_report, encoding="utf-8")
    print(f"Size report saved to {SIZE_REPORT_PATH}")

if __name__ == "__main__":
    main()
