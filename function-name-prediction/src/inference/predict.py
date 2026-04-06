import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer
from src.features.vectorizer import normalize_text, transform_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TFLITE_MODEL_PATH = PROJECT_ROOT / "models" / "function_model.tflite"
H5_MODEL_PATH = PROJECT_ROOT / "models" / "function_model.h5"
WORD_INDEX_PATH = PROJECT_ROOT / "models" / "word_index.pkl"
OUTPUT_TOKENIZER_PATH = PROJECT_ROOT / "models" / "output_tokenizer.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
INFERENCE_SPEED_REPORT_PATH = REPORTS_DIR / "inference_speed.txt"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

_interpreter = None
_input_details = None
_output_details = None
_keras_model = None
_word_index = None
_output_tokenizer = None


def to_camel_case(words: List[str]) -> str:
    if not words:
        return ""
    return words[0] + "".join(w.capitalize() for w in words[1:])


def resources_loaded() -> bool:
    return _word_index is not None and _output_tokenizer is not None and (_interpreter is not None or _keras_model is not None)


def load_resources():
    global _interpreter, _input_details, _output_details, _keras_model, _word_index, _output_tokenizer

    if resources_loaded():
        return

    if not WORD_INDEX_PATH.exists() or not OUTPUT_TOKENIZER_PATH.exists():
        raise FileNotFoundError("Tokenizer artifacts not found. Please run: python run_pipeline.py")

    with WORD_INDEX_PATH.open("rb") as f:
        _word_index = pickle.load(f)
    with OUTPUT_TOKENIZER_PATH.open("rb") as f:
        _output_tokenizer = pickle.load(f)

    if TFLITE_MODEL_PATH.exists():
        try:
            _interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
            _interpreter.allocate_tensors()
            _input_details = _interpreter.get_input_details()
            _output_details = _interpreter.get_output_details()
        except Exception:
            # Fallback to Keras model when TFLite model requires unsupported runtime ops.
            _interpreter = None
            if H5_MODEL_PATH.exists():
                _keras_model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
            else:
                raise
    elif H5_MODEL_PATH.exists():
        # Some H5 models may reference a custom serialized layer named 'NotEqual'.
        # Provide a minimal compatible implementation so Keras can deserialize it.
        class NotEqual(Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def __call__(self, *args, **kwargs):
                new_args = []
                for a in args:
                    if tf.is_tensor(a):
                        new_args.append(a)
                    else:
                        try:
                            new_args.append(tf.convert_to_tensor(a))
                        except Exception:
                            new_args.append(a)
                return super().__call__(*new_args, **kwargs)

            def call(self, inputs):
                if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                    a, b = inputs
                else:
                    raise ValueError("NotEqual layer expects a list/tuple of two tensors")
                return tf.not_equal(a, b)

            def get_config(self):
                return super().get_config()

        _keras_model = tf.keras.models.load_model(H5_MODEL_PATH, custom_objects={"NotEqual": NotEqual})
    else:
        raise FileNotFoundError("Model file not found. Please run: python run_pipeline.py")


def decode_tokens(token_ids: List[int], output_index_word: Dict[int, str]) -> List[str]:
    words = []
    for idx in token_ids:
        token = output_index_word.get(int(idx), UNK_TOKEN)
        if token in {PAD_TOKEN, UNK_TOKEN}:
            continue
        words.append(token)
    return words


def run_inference(input_ids: np.ndarray) -> np.ndarray:
    if _interpreter is not None:
        input_dtype = _input_details[0]["dtype"]
        model_input = input_ids.astype(input_dtype, copy=False)
        _interpreter.set_tensor(_input_details[0]["index"], model_input)
        _interpreter.invoke()
        return _interpreter.get_tensor(_output_details[0]["index"])

    return _keras_model.predict(input_ids, verbose=0)


def predict_function(metadata_text: str) -> str:
    load_resources()

    normalized_text = normalize_text(metadata_text)
    max_input_len = int(_output_tokenizer["max_input_len"])
    input_seq = transform_text([normalized_text], _word_index, max_input_len)

    probs = run_inference(input_seq)
    pred_ids = np.argmax(probs, axis=-1)[0].tolist()

    output_index_word = _output_tokenizer["output_index_word"]
    words = decode_tokens(pred_ids, output_index_word)
    predicted_name = to_camel_case(words)
    return predicted_name if predicted_name else "unknownFunction"


def benchmark_inference(sample_text: str):
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
    print("\n--- Function Name Predictor (TFLite Inference) ---")
    example_input = "Adds two integers int a int b return int keywords add sum"
    print(f"\nInput Metadata: \"{example_input}\"")
    try:
        predicted_name, latency_ms = benchmark_inference(example_input)
        print(f"Predicted Function: {predicted_name}")
        print(f"Inference latency: {latency_ms:.4f} ms")
        print(f"Speed report saved to: {INFERENCE_SPEED_REPORT_PATH}")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Error: {e}")
