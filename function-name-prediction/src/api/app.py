import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add the project root to sys.path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.inference.predict import predict_function, load_resources

# Define the request body schema
class PredictRequest(BaseModel):
    metadata: str

# Define the response schema
class PredictResponse(BaseModel):
    predicted_function: str

# Initialize the FastAPI app
app = FastAPI(
    title="Function Name Predictor API",
    description="A lightweight API that predicts a function name from structured function metadata.",
    version="1.0.0"
)

# Load the models when the application starts
@app.on_event("startup")
async def startup_event():
    print("Starting up Function Name Predictor API...")
    try:
        load_resources()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")
        # In a real production app, we might want to log this and potentially stop the app,
        # but for this lightweight version, we'll let it handle errors during the request.

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict a function name based on the provided metadata string.
    """
    try:
        # Input validation
        if not request.metadata or not request.metadata.strip():
            raise HTTPException(status_code=400, detail="Metadata string cannot be empty.")
            
        # Run prediction
        predicted_name = predict_function(request.metadata)
        
        # Return result
        return PredictResponse(predicted_function=predicted_name)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"Model files missing. Ensure the model has been trained. {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "Function Name Predictor API is running."}

if __name__ == "__main__":
    import uvicorn
    # Make sure we use the correct path or module reference depending on from where it's run
    print("Starting API with uvicorn...")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
