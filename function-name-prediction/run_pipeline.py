"""
run_pipeline.py

Main orchestrator for the Function Name Prediction ML pipeline.
Pipeline stages:
1) Data generation
2) Metadata preprocessing
3) FastText embedding training + BiGRU model training
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
from src.models.convert_to_tflite import convert_model_to_tflite

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
        df_raw = generate_dataset(num_records=720, random_state=42)
        df_raw.to_csv(raw_data_path, index=False)
        logging.info(f"Dataset generated and saved to {raw_data_path}")

        # Step 2: Preprocessing
        logging.info("Step 2: Preprocessing metadata...")
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        df_processed = prepare_dataset(raw_data_path)
        save_processed_data(df_processed, processed_data_path)
        logging.info(f"Preprocessed dataset saved to {processed_data_path}")
        
        # Step 3: Train FastText features and TensorFlow BiGRU model
        logging.info("Step 3: Training FastText + TensorFlow BiGRU model...")
        train_model_main()

        # Step 4: Convert trained Keras model to optimized TensorFlow Lite
        logging.info("Step 4: Converting Keras model to TensorFlow Lite...")
        tflite_path = convert_model_to_tflite()
        logging.info(f"TFLite model saved to {tflite_path}")

        logging.info("Pipeline executed successfully. All models and artifacts saved.")
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
