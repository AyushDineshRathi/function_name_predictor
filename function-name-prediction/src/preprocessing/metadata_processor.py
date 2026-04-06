import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = [
    "description",
    "parameters",
    "return_type",
    "library",
    "keywords",
    "function_name",
]


def _normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _to_text(value) -> str:
    """Convert values safely to text, replacing null-like values with an empty string."""
    if pd.isna(value):
        return ""
    return str(value)


def clean_text(value) -> str:
    """
    Generic cleaner:
    - lowercase
    - remove punctuation/special chars
    - normalize extra spaces
    """
    text = _to_text(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return _normalize_spaces(text)


def clean_parameters(value) -> str:
    """
    Parameters cleaner:
    - lowercase
    - remove type syntax symbols such as < > ,
    - remove remaining punctuation
    - normalize spaces
    """
    text = _to_text(value).lower()
    text = re.sub(r"[<>,]", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return _normalize_spaces(text)


def clean_keywords(value) -> str:
    """
    Keywords cleaner:
    - split by comma
    - clean each keyword token
    - join with spaces
    """
    text = _to_text(value).lower()
    if not text:
        return ""

    parts = text.split(",")
    cleaned_parts = []
    for part in parts:
        token = re.sub(r"[^a-z0-9\s]", " ", part)
        token = _normalize_spaces(token)
        if token:
            cleaned_parts.append(token)
    return " ".join(cleaned_parts)


def build_input_text(description: str, parameters: str, return_type: str, library: str, keywords: str) -> str:
    """Build model input in the exact tagged format requested."""
    return (
        f"[DESC] {description} "
        f"[PARAMS] {parameters} "
        f"[RET] {return_type} "
        f"[LIB] {library} "
        f"[KEY] {keywords}"
    )


def prepare_dataset(path: str) -> pd.DataFrame:
    """
    Load raw data and output DataFrame with required columns:
    - input_text
    - function_name
    """
    input_path = Path(path)
    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Cleaning and formatting metadata...")
    description = df["description"].apply(clean_text)
    parameters = df["parameters"].apply(clean_parameters)
    return_type = df["return_type"].apply(clean_text)
    library = df["library"].apply(clean_text)
    keywords = df["keywords"].apply(clean_keywords)

    input_text = [
        build_input_text(d, p, r, l, k)
        for d, p, r, l, k in zip(description, parameters, return_type, library, keywords)
    ]

    result_df = pd.DataFrame(
        {
            "input_text": input_text,
            "function_name": df["function_name"].fillna("").astype(str),
        }
    )

    print("Dataset preparation complete.")
    return result_df


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save processed dataset to disk."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving processed dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Save complete.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input/output CSV paths."""
    default_raw = PROJECT_ROOT / "data" / "raw" / "function_dataset.csv"
    fallback_raw = PROJECT_ROOT / "data" / "raw" / "functions_dataset.csv"
    default_input = default_raw if default_raw.exists() else fallback_raw
    default_output = PROJECT_ROOT / "data" / "processed" / "processed_dataset.csv"

    parser = argparse.ArgumentParser(description="Preprocess function metadata for NLP training.")
    parser.add_argument("--input", type=str, default=str(default_input), help="Path to input CSV file.")
    parser.add_argument("--output", type=str, default=str(default_output), help="Path to output CSV file.")
    return parser.parse_args()


def main() -> None:
    """Run metadata preprocessing end-to-end."""
    args = parse_args()
    processed_df = prepare_dataset(args.input)
    save_processed_data(processed_df, args.output)

    if not processed_df.empty:
        print("\nExample output row:")
        print(processed_df.iloc[0]["input_text"])
        print(f"-> {processed_df.iloc[0]['function_name']}")


if __name__ == "__main__":
    main()
