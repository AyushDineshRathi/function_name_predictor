import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "function_model.pkl"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_REPORT_PATH = REPORTS_DIR / "model_metrics.txt"
SIZE_REPORT_PATH = REPORTS_DIR / "model_size.txt"

# Add the project root to sys.path so we can import src modules
sys.path.append(str(PROJECT_ROOT))

from src.features.vectorizer import train_vectorizer, transform_text, save_vectorizer

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
    
    X_text = df['combined_metadata'].astype(str)
    y = df['function_name']
    
    # 2. Split data into train/test (80/20)
    print("\nSplitting data into train/test (80/20)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

    # 3. Train TF-IDF vectorizer on train split only, then transform test split
    X_train = train_vectorizer(X_train_text)
    X_test = transform_text(X_test_text)

    # Save vectorizer
    save_vectorizer(str(VECTORIZER_PATH))
    
    print(f"\nTraining set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # 4. Train Models
    print("\nTraining models...")
    
    # Logistic Regression
    lr_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_metrics = evaluate_model(y_test, lr_preds, "Logistic Regression")
    
    # Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_preds = nb_model.predict(X_test)
    nb_metrics = evaluate_model(y_test, nb_preds, "Multinomial Naive Bayes")
    
    # 5. Select Best Model
    print("\n==================================")
    print("Model Selection:")
    
    best_model = None
    best_name = ""
    
    if lr_metrics["f1"] >= nb_metrics["f1"]:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        best_model = nb_model
        best_name = "Multinomial Naive Bayes"
        
    print(f"Best model selected: {best_name} (F1 Score: {max(lr_metrics['f1'], nb_metrics['f1']):.4f})")
    print("==================================")
    
    # 6. Save the best model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving best model to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Model saved successfully. Training pipeline complete.")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_report = (
        "Model Evaluation Report\n"
        "=======================\n\n"
        f"{_format_metrics_block(lr_metrics)}\n"
        f"{_format_metrics_block(nb_metrics)}\n"
        f"Best model: {best_name}\n"
    )
    METRICS_REPORT_PATH.write_text(metrics_report, encoding="utf-8")
    print(f"Metrics report saved to {METRICS_REPORT_PATH}")

    model_size_kb = MODEL_PATH.stat().st_size / 1024
    vectorizer_size_kb = VECTORIZER_PATH.stat().st_size / 1024
    print(f"Model size (function_model.pkl): {model_size_kb:.2f} KB")
    print(f"Vectorizer size (vectorizer.pkl): {vectorizer_size_kb:.2f} KB")

    size_report = (
        "Model Artifact Sizes\n"
        "====================\n"
        f"function_model.pkl: {model_size_kb:.2f} KB\n"
        f"vectorizer.pkl: {vectorizer_size_kb:.2f} KB\n"
    )
    SIZE_REPORT_PATH.write_text(size_report, encoding="utf-8")
    print(f"Size report saved to {SIZE_REPORT_PATH}")

if __name__ == "__main__":
    main()
