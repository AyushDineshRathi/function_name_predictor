import argparse
import pandas as pd
from pathlib import Path
from src.preprocessing.text_normalizer import build_structured_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def prepare_dataset(path: str) -> pd.DataFrame:
    """
    Load raw data and output DataFrame with required columns:
    - input_text
    - function_name
    """
    input_path = Path(path)
    print(f"Loading raw dataset from {input_path}...")
    df = pd.read_csv(input_path)
    
    print("Combining metadata...")
    # Combine metadata into a single structured feature using format:
    # "desc:add two integers | params:int int | return:int | keywords:add sum"
    
    def combine_row(row):
        return build_structured_metadata(
            description=row.get("description", ""),
            parameters=row.get("parameters", ""),
            return_type=row.get("return_type", ""),
            keywords=row.get("keywords", ""),
        )
        
    df['combined_metadata'] = df.apply(combine_row, axis=1)
    
    # Keep only the necessary columns
    result_df = df[['combined_metadata', 'function_name']]
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
    default_input = PROJECT_ROOT / "data" / "raw" / "functions_dataset.csv"
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
        print(processed_df.iloc[0]["combined_metadata"])
        print(f"-> {processed_df.iloc[0]['function_name']}")


if __name__ == "__main__":
    main()
