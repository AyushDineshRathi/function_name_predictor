import pandas as pd
import re
import os

def clean_text(text):
    """
    Cleans text by converting to lowercase, removing unnecessary punctuation,
    and normalizing whitespace.
    """
    if pd.isna(text):
        return ""
    
    # Convert to string to be safe
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep letters, numbers, and spaces)
    # Using regex to replace anything that isn't a word character or whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Normalize whitespace (replace multiple spaces with a single space)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_dataset(path: str) -> pd.DataFrame:
    """
    Loads raw dataset, cleans text fields, and combines metadata into a single text feature.
    
    Outputs a DataFrame with columns: ['combined_metadata', 'function_name']
    """
    print(f"Loading raw dataset from {path}...")
    df = pd.read_csv(path)
    
    print("Cleaning text fields...")
    # Clean relevant text columns
    text_columns = ['description', 'parameters', 'return_type', 'library', 'keywords']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
            
    print("Combining metadata...")
    # Combine metadata into a single text feature using the requested format
    # "description parameters return return_type library library_name keywords keyword_list params param_count"
    
    def combine_row(row):
        parts = []
        
        if pd.notna(row.get('description')) and row['description']:
            parts.append(row['description'])
            
        if pd.notna(row.get('parameters')) and row['parameters']:
            parts.append(row['parameters'])
            
        if pd.notna(row.get('return_type')) and row['return_type']:
            parts.append(f"return {row['return_type']}")
            
        if pd.notna(row.get('library')) and row['library']:
            parts.append(f"library {row['library']}")
            
        if pd.notna(row.get('keywords')) and row['keywords']:
            parts.append(f"keywords {row['keywords']}")
            
        if pd.notna(row.get('param_count')):
            parts.append(f"params {row['param_count']}")
            
        return " ".join(parts)
        
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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Saving processed dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Save complete.")

if __name__ == "__main__":
    # For testing the module directly
    raw_path = os.path.join("data", "raw", "functions_dataset.csv")
    processed_path = os.path.join("data", "processed", "processed_dataset.csv")
    
    if os.path.exists(raw_path):
        processed_df = prepare_dataset(raw_path)
        
        # Display an example of the combined text
        print("\nExample combined metadata:")
        print(processed_df['combined_metadata'].iloc[0])
        print(f"-> Target: {processed_df['function_name'].iloc[0]}\n")
        
        save_processed_data(processed_df, processed_path)
    else:
        print(f"Error: Raw dataset not found at {raw_path}")
