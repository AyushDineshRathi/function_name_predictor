import os
import joblib

# Global variables to cache the model and vectorizer
_model = None
_vectorizer = None

def load_resources(model_path=None, vectorizer_path=None):
    """
    Loads the trained model and vectorizer from disk.
    Cached after the first call to avoid reloading on subsequent predictions.
    """
    global _model, _vectorizer
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if model_path is None:
        model_path = os.path.join(base_dir, "models", "function_model.pkl")
    if vectorizer_path is None:
        vectorizer_path = os.path.join(base_dir, "models", "vectorizer.pkl")
        
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        _model = joblib.load(model_path)
        
    if _vectorizer is None:
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}. Please train the model first.")
        _vectorizer = joblib.load(vectorizer_path)

def predict_function(metadata_text: str) -> str:
    """
    Transforms the input metadata text and returns the predicted function name.
    
    Args:
        metadata_text (str): The combined and processed metadata string.
        
    Returns:
        str: The predicted function name.
    """
    # Ensure models are loaded
    load_resources()
    
    # 1. Transform input metadata text using vectorizer
    # The vectorizer expects an iterable of strings
    transformed_text = _vectorizer.transform([metadata_text])
    
    # 2. Run model prediction
    prediction = _model.predict(transformed_text)
    
    # 3. Return predicted function name
    return prediction[0]

if __name__ == "__main__":
    import sys
    
    # Ensure standard paths resolve correctly inside the script too
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    
    print("\n--- Function Name Predictor (Inference Demonstration) ---")
    
    # Example input as specified
    example_input = "adds two integers int a int b return int keywords add sum"
    print(f"\nInput Metadata: \"{example_input}\"")
    
    try:
        predicted_name = predict_function(example_input)
        print(f"Expected output example:")
        print(f"Predicted Function: {predicted_name}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
