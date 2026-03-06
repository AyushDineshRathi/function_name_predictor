import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VECTORIZER_PATH = PROJECT_ROOT / "models" / "vectorizer.pkl"

# Global vectorizer instance
_vectorizer = None

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    return _vectorizer

def set_vectorizer(vectorizer):
    global _vectorizer
    _vectorizer = vectorizer

def train_vectorizer(text_data):
    """
    Fits the TF-IDF vectorizer on the provided text data.
    
    Args:
        text_data (list or pd.Series): The text data to train on.
        
    Returns:
        scipy.sparse matrix: The transformed text data.
    """
    vec = get_vectorizer()
    print("Training TF-IDF vectorizer on data...")
    transformed_data = vec.fit_transform(text_data)
    print(f"Vectorization complete. Vocabulary size: {len(vec.vocabulary_)}")
    return transformed_data

def transform_text(text_data):
    """
    Transforms new text data using the already fitted vectorizer.
    
    Args:
        text_data (list or pd.Series): The text data to transform.
        
    Returns:
        scipy.sparse matrix: The transformed text data.
    """
    vec = get_vectorizer()
    if not hasattr(vec, 'vocabulary_'):
        raise ValueError("Vectorizer has not been trained yet. Call train_vectorizer or load_vectorizer first.")
        
    return vec.transform(text_data)

def save_vectorizer(path=DEFAULT_VECTORIZER_PATH):
    """
    Saves the trained vectorizer to disk using joblib.
    
    Args:
        path (str): The file path to save to. Defaults to 'models/vectorizer.pkl'.
    """
    vec = get_vectorizer()
    if not hasattr(vec, 'vocabulary_'):
        print("Warning: Saving an untrained vectorizer.")
        
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving vectorizer to {target_path}...")
    joblib.dump(vec, target_path)
    print("Vectorizer saved successfully.")

def load_vectorizer(path=DEFAULT_VECTORIZER_PATH):
    """
    Loads a trained vectorizer from disk using joblib.
    
    Args:
        path (str): The file path to load from. Defaults to 'models/vectorizer.pkl'.
        
    Returns:
        TfidfVectorizer: The loaded vectorizer instance.
    """
    vectorizer_path = Path(path)
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

    print(f"Loading vectorizer from {vectorizer_path}...")
    loaded_vec = joblib.load(vectorizer_path)
    set_vectorizer(loaded_vec)
    print("Vectorizer loaded successfully.")
    return loaded_vec

if __name__ == "__main__":
    # Small test
    sample_texts = [
        "adds two integers int a int b return int library mathutils keywords add sum params 2",
        "reads all content from a file string filepath return string library fileutils keywords read file io params 1"
    ]
    
    transformed = train_vectorizer(sample_texts)
    print(f"Transformed shape: {transformed.shape}")
    
    save_vectorizer(DEFAULT_VECTORIZER_PATH)
    
    # Test loading
    load_vectorizer(DEFAULT_VECTORIZER_PATH)
    test_transform = transform_text(["new test string return int params 0"])
    print(f"Test transform shape: {test_transform.shape}")
