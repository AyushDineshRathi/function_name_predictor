from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"

_vectorizer = None


def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2,
            stop_words="english",
        )
    return _vectorizer


def set_vectorizer(vectorizer):
    global _vectorizer
    _vectorizer = vectorizer


def train_vectorizer(text_data):
    vec = get_vectorizer()
    print("Training TF-IDF vectorizer on data...")
    transformed = vec.fit_transform(text_data)
    print(f"Vectorization complete. Vocabulary size: {len(vec.vocabulary_)}")
    return transformed


def transform_text(text_data):
    vec = get_vectorizer()
    if not hasattr(vec, "vocabulary_"):
        raise ValueError("Vectorizer has not been trained yet. Call train_vectorizer or load_vectorizer first.")
    return vec.transform(text_data)


def save_vectorizer(path=DEFAULT_VECTORIZER_PATH):
    vec = get_vectorizer()
    if not hasattr(vec, "vocabulary_"):
        print("Warning: saving an untrained vectorizer.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, target)
    print(f"Vectorizer saved to {target}")


def load_vectorizer(path=DEFAULT_VECTORIZER_PATH):
    vectorizer_path = Path(path)
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")
    loaded_vec = joblib.load(vectorizer_path)
    set_vectorizer(loaded_vec)
    print(f"Vectorizer loaded from {vectorizer_path}")
    return loaded_vec
