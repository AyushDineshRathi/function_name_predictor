import pandas as pd
from pathlib import Path
from src.preprocessing.text_normalizer import build_structured_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def prepare_dataset(path: str) -> pd.DataFrame:
    """
    Loads raw dataset, cleans text fields, and combines metadata into a single text feature.
    
    Outputs a DataFrame with columns: ['combined_metadata', 'function_name']
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

def save_processed_data(df: pd.DataFrame, output_path: str):
    """
    Saves the processed dataset to the specified path.
    """
    # Ensure directory exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving processed dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Save complete.")

if __name__ == "__main__":
    # For testing the module directly
    raw_path = PROJECT_ROOT / "data" / "raw" / "functions_dataset.csv"
    processed_path = PROJECT_ROOT / "data" / "processed" / "processed_dataset.csv"
    
    if raw_path.exists():
        processed_df = prepare_dataset(raw_path)
        
        # Display an example of the combined text
        print("\nExample combined metadata:")
        print(processed_df['combined_metadata'].iloc[0])
        print(f"-> Target: {processed_df['function_name'].iloc[0]}\n")
        
        save_processed_data(processed_df, processed_path)
    else:
        print(f"Error: Raw dataset not found at {raw_path}")
