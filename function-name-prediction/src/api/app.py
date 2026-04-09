import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add the project root to sys.path so we can import src modules
sys.path.append(str(PROJECT_ROOT))

from src.inference.predict import predict_with_confidence, load_resources, resources_loaded

# Define the request body schema
class PredictRequest(BaseModel):
    metadata: str

# Define the response schema
class PredictResponse(BaseModel):
    predicted_function: str
    best_label: str
    best_confidence: float
    threshold: float
    top_predictions: List[dict]

@asynccontextmanager
async def lifespan(_: FastAPI):
    print("Starting up Function Name Predictor API...")
    load_resources()
    print("Models loaded successfully.")
    yield

# Initialize the FastAPI app
app = FastAPI(
    title="Function Name Predictor API",
    description="A lightweight API that predicts a function name from structured function metadata.",
    version="1.0.0",
    lifespan=lifespan,
)

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
        prediction = predict_with_confidence(request.metadata, top_k=3)
        
        # Return result
        return PredictResponse(**prediction)
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Models not found. Please run: python run_pipeline.py")
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint with model/vectorizer readiness.
    """
    return {"status": "ok", "model_loaded": resources_loaded()}

if __name__ == "__main__":
    import uvicorn
    print("Starting API with uvicorn...")
    print("Open API docs at: http://127.0.0.1:8000/docs")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)
