from pathlib import Path

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
H5_MODEL_PATH = PROJECT_ROOT / "models" / "function_model.h5"
TFLITE_MODEL_PATH = PROJECT_ROOT / "models" / "function_model.tflite"


def convert_model_to_tflite(
    h5_model_path: Path = H5_MODEL_PATH,
    tflite_model_path: Path = TFLITE_MODEL_PATH,
) -> Path:
    """Convert Keras .h5 model to optimized TensorFlow Lite."""
    if not h5_model_path.exists():
        raise FileNotFoundError(f"Keras model not found at {h5_model_path}. Train the model first.")

    model = tf.keras.models.load_model(h5_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    tflite_model_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_model_path.write_bytes(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")
    return tflite_model_path


def main():
    convert_model_to_tflite()


if __name__ == "__main__":
    main()
