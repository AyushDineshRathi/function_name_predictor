"""
run_pipeline.py

Main orchestrator for the Function Name Prediction ML pipeline.
This script binds together data ingestion, preprocessing, training, and output.
"""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Ensure project modules can be loaded
sys.path.append(str(PROJECT_ROOT))

from src.data.dataset_generator import generate_dataset
from src.preprocessing.metadata_processor import prepare_dataset, save_processed_data
from src.models.train_model import main as train_model_main

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    logging.info("Starting Function Name Prediction pipeline...")

    # Define paths
    raw_data_dir = PROJECT_ROOT / "data" / "raw"
    raw_data_path = raw_data_dir / "functions_dataset.csv"
    
    processed_data_dir = PROJECT_ROOT / "data" / "processed"
    processed_data_path = processed_data_dir / "processed_dataset.csv"

    try:
        # Step 1: Generate dataset
        logging.info("Step 1: Generating synthetic dataset...")
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        df_raw = generate_dataset(num_records=720)
        df_raw.to_csv(raw_data_path, index=False)
        logging.info(f"Dataset generated and saved to {raw_data_path}")

        # Step 2: Preprocessing
        logging.info("Step 2: Preprocessing metadata...")
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        df_processed = prepare_dataset(raw_data_path)
        save_processed_data(df_processed, processed_data_path)
        logging.info(f"Preprocessed dataset saved to {processed_data_path}")

        # Step 3, 4, 5, 6: Feature Extraction, Training, Evaluation, Saving Models
        logging.info("Step 3-6: Training vectorizer and models...")
        # train_model_main() naturally handles vectorization, training both models,
        # evaluating them, printing metrics, picking the best one, and saving to disk.
        train_model_main()

        logging.info("Pipeline executed successfully. All models and artifacts saved.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
