import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from gensim.models import FastText

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MATRIX_PATH = PROJECT_ROOT / "models" / "embedding_matrix.npy"
WORD_INDEX_PATH = PROJECT_ROOT / "models" / "word_index.pkl"

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            min_df=2,
            stop_words='english',
        )
    return _vectorizer


def normalize_text(text: str) -> str:
    """Lowercase, remove punctuation, and normalize spaces."""
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_text(text: str) -> List[str]:
    """Tokenize by whitespace split, as required."""
    return normalize_text(text).split()


def texts_to_token_lists(text_data) -> List[List[str]]:
    """Convert iterable text data to tokenized sentences."""
    return [tokenize_text(text) for text in text_data]


def build_word_index(tokenized_texts: List[List[str]], max_vocab_size: int = 12000) -> Dict[str, int]:
    """Build input word index with reserved PAD/UNK tokens."""
    token_counter = Counter(token for sent in tokenized_texts for token in sent)
    most_common = token_counter.most_common(max_vocab_size)

    word_index = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in most_common:
        if token not in word_index:
            word_index[token] = len(word_index)
    return word_index


def train_fasttext(tokenized_texts: List[List[str]], vector_size: int = 100, window: int = 5, min_count: int = 1) -> FastText:
    """Train gensim FastText embeddings from tokenized input text."""
    print("Training FastText embeddings...")
    model = FastText(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
    )
    return model


def build_embedding_matrix(word_index: Dict[str, int], ft_model: FastText, vector_size: int = 100) -> np.ndarray:
    """Create embedding matrix aligned to word_index."""
    matrix = np.zeros((len(word_index), vector_size), dtype=np.float32)
    rng = np.random.default_rng(42)

    for token, idx in word_index.items():
        if token == PAD_TOKEN:
            continue
        if token in ft_model.wv:
            matrix[idx] = ft_model.wv[token]
        else:
            matrix[idx] = rng.normal(0.0, 0.05, vector_size)
    return matrix


def texts_to_sequences(text_data, word_index: Dict[str, int]) -> List[List[int]]:
    """Map tokenized input text to integer ids."""
    unk_idx = word_index[UNK_TOKEN]
    tokenized = texts_to_token_lists(text_data)
    return [[word_index.get(tok, unk_idx) for tok in sent] for sent in tokenized]


def pad_sequences(sequences: List[List[int]], max_len: int, pad_value: int = 0) -> np.ndarray:
    """Pad/truncate variable-length sequences to fixed length."""
    padded = np.full((len(sequences), max_len), pad_value, dtype=np.int64)
    for i, seq in enumerate(sequences):
        trunc = seq[:max_len]
        padded[i, : len(trunc)] = trunc
    return padded


def train_vectorizer(text_data, max_vocab_size: int = 12000) -> Tuple[np.ndarray, Dict[str, int], np.ndarray, int]:
    """
    End-to-end input vectorization:
    - tokenize
    - train FastText
    - build word index
    - build embedding matrix
    - convert to padded sequences
    """
    tokenized = texts_to_token_lists(text_data)
    word_index = build_word_index(tokenized, max_vocab_size=max_vocab_size)
    ft_model = train_fasttext(tokenized, vector_size=100, window=5, min_count=1)
    embedding_matrix = build_embedding_matrix(word_index, ft_model, vector_size=100)

    sequences = texts_to_sequences(text_data, word_index)
    max_len = max(8, min(80, int(np.percentile([len(s) for s in sequences], 95)) if sequences else 8))
    padded = pad_sequences(sequences, max_len=max_len, pad_value=word_index[PAD_TOKEN])

    print(f"Vectorization complete. Vocab size: {len(word_index)} | Max input length: {max_len}")
    return padded, word_index, embedding_matrix, max_len


def transform_text(text_data, word_index: Dict[str, int], max_len: int) -> np.ndarray:
    """Transform new text data into padded index sequences."""
    sequences = texts_to_sequences(text_data, word_index)
    return pad_sequences(sequences, max_len=max_len, pad_value=word_index[PAD_TOKEN])


def save_vectorizer(embedding_matrix: np.ndarray, word_index: Dict[str, int]) -> None:
    """Save embedding matrix (npy) and word index (pickle)."""
    EMBEDDING_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDING_MATRIX_PATH, embedding_matrix)
    with WORD_INDEX_PATH.open("wb") as f:
        pickle.dump(word_index, f)
    print(f"Saved embedding matrix to {EMBEDDING_MATRIX_PATH}")
    print(f"Saved word index to {WORD_INDEX_PATH}")


def load_vectorizer():
    """Load embedding matrix and word index from disk."""
    if not EMBEDDING_MATRIX_PATH.exists() or not WORD_INDEX_PATH.exists():
        raise FileNotFoundError("Embedding artifacts missing. Please run: python run_pipeline.py")

    embedding_matrix = np.load(EMBEDDING_MATRIX_PATH)
    with WORD_INDEX_PATH.open("rb") as f:
        word_index = pickle.load(f)
    return embedding_matrix, word_index
