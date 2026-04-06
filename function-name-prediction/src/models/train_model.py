import pickle
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.features.vectorizer import save_vectorizer, train_vectorizer, transform_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_dataset.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "function_model.h5"
OUTPUT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "output_tokenizer.pkl"
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


def encode_output_sequences(function_names: pd.Series, output_word_index: Dict[str, int], max_output_len: int) -> np.ndarray:
    """Encode function names as fixed-length output token ids."""
    unk_idx = output_word_index[UNK_TOKEN]
    pad_idx = output_word_index[PAD_TOKEN]
    encoded = np.full((len(function_names), max_output_len), pad_idx, dtype=np.int32)

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
        mask_zero=True,
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

    X_train_vec, word_index, embedding_matrix, max_input_len = train_vectorizer(X_train_text, max_vocab_size=10000)
    save_vectorizer(embedding_matrix, word_index)
    X_test_vec = transform_text(X_test_text, word_index, max_input_len)

    output_tokenizer = build_output_tokenizer(y_train_fn, max_output_len=4)
    output_word_index = output_tokenizer["output_word_index"]
    max_output_len = output_tokenizer["max_output_len"]
    y_train = encode_output_sequences(y_train_fn, output_word_index, max_output_len)
    y_test = encode_output_sequences(y_test_fn, output_word_index, max_output_len)

    model = build_bigru_model(
        embedding_matrix=embedding_matrix.astype(np.float32),
        max_input_len=max_input_len,
        output_vocab_size=len(output_word_index),
        max_output_len=max_output_len,
        hidden_size=96,
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
