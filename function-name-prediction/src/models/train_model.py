import pickle
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
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

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def split_camel_case(name: str) -> List[str]:
    """Convert camelCase/PascalCase function names into lowercase word tokens."""
    if not name:
        return []
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name)).split()
    return [p.lower() for p in parts]


def build_output_tokenizer(function_names: pd.Series, max_output_len: int = 4) -> Dict:
    """Build mappings for output token sequences."""
    token_lists = [split_camel_case(fn) for fn in function_names.fillna("").astype(str)]
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tokens in token_lists:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    return {
        "output_word_index": vocab,
        "output_index_word": inv_vocab,
        "max_output_len": max_output_len,
    }


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

    for i, fn in enumerate(function_names.fillna("").astype(str)):
        tokens = split_camel_case(fn)[:max_output_len]
        token_ids = [output_word_index.get(tok, unk_idx) for tok in tokens]
        encoded[i, : len(token_ids)] = token_ids
    return encoded


def build_bigru_model(
    embedding_matrix: np.ndarray,
    max_input_len: int,
    output_vocab_size: int,
    max_output_len: int,
    hidden_size: int = 96,
) -> tf.keras.Model:
    """
    Embedding (FastText weights) -> BiGRU -> Dense -> reshape -> softmax.
    """
    vocab_size, embedding_dim = embedding_matrix.shape

    inputs = tf.keras.Input(shape=(max_input_len,), dtype=tf.int32, name="input_ids")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True,
        mask_zero=False,
        name="embedding",
    )(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(hidden_size, return_sequences=False),
        name="bigru",
    )(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout")(x)
    x = tf.keras.layers.Dense(max_output_len * output_vocab_size, name="dense_logits")(x)
    x = tf.keras.layers.Reshape((max_output_len, output_vocab_size), name="reshape_logits")(x)
    outputs = tf.keras.layers.Softmax(axis=-1, name="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="function_name_bigru")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="token_accuracy")],
    )
    return model


def exact_match_accuracy(y_true: np.ndarray, y_pred_probs: np.ndarray) -> float:
    y_pred = np.argmax(y_pred_probs, axis=-1)
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def load_data(path=PROCESSED_DATA_PATH) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {data_path}")
    print(f"Loading processed dataset from {data_path}...")
    return pd.read_csv(data_path)


def main():
    tf.random.set_seed(42)
    np.random.seed(42)

    df = load_data().dropna(subset=["input_text", "function_name"])

    X_text = df["input_text"].astype(str)
    y_fn = df["function_name"].astype(str)

    X_train_text, X_test_text, y_train_fn, y_test_fn = train_test_split(
        X_text, y_fn, test_size=0.2, random_state=42
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

    output_tokenizer = build_output_tokenizer(y_train_fn, max_output_len=4)
    output_word_index = output_tokenizer["output_word_index"]
    max_output_len = output_tokenizer["max_output_len"]
    y_train = encode_output_sequences(y_train_fn, output_word_index, max_output_len)
    y_test = encode_output_sequences(y_test_fn, output_word_index, max_output_len)

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

    print(model.summary())
    history = model.fit(
        X_train_vec,
        y_train,
        validation_data=(X_test_vec, y_test),
        epochs=8,
        batch_size=32,
        verbose=1,
    )

    test_pred_probs = model.predict(X_test_vec, batch_size=64, verbose=0)
    test_exact = exact_match_accuracy(y_test, test_pred_probs)
    print(f"Final test exact-match accuracy: {test_exact:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

    output_tokenizer["max_input_len"] = int(max_input_len)
    with OUTPUT_TOKENIZER_PATH.open("wb") as f:
        pickle.dump(output_tokenizer, f)
    print(f"Saved output tokenizer to {OUTPUT_TOKENIZER_PATH}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_report = ["BiGRU (TensorFlow) Training Report", "=================================", ""]
    for idx in range(len(history.history.get("loss", []))):
        loss_val = history.history["loss"][idx]
        val_loss = history.history.get("val_loss", [None] * len(history.history["loss"]))[idx]
        tok_acc = history.history.get("token_accuracy", [None] * len(history.history["loss"]))[idx]
        val_tok_acc = history.history.get("val_token_accuracy", [None] * len(history.history["loss"]))[idx]
        metrics_report.append(
            f"Epoch {idx + 1}: loss={loss_val:.4f}, val_loss={val_loss:.4f}, "
            f"token_acc={tok_acc:.4f}, val_token_acc={val_tok_acc:.4f}"
        )
    metrics_report.append(f"\nFinal test exact-match accuracy: {test_exact:.4f}")
    METRICS_REPORT_PATH.write_text("\n".join(metrics_report), encoding="utf-8")

    model_size_kb = MODEL_PATH.stat().st_size / 1024
    embed_size_kb = (PROJECT_ROOT / "models" / "embedding_matrix.npy").stat().st_size / 1024
    tok_size_kb = OUTPUT_TOKENIZER_PATH.stat().st_size / 1024
    size_report = (
        "Model Size Report\n"
        "=================\n"
        f"function_model.h5 : {model_size_kb:.2f} KB\n"
        f"embedding_matrix.npy : {embed_size_kb:.2f} KB\n"
        f"output_tokenizer.pkl : {tok_size_kb:.2f} KB\n"
    )
    SIZE_REPORT_PATH.write_text(size_report, encoding="utf-8")
    print(f"Reports written to {METRICS_REPORT_PATH} and {SIZE_REPORT_PATH}")


if __name__ == "__main__":
    main()
