# Code with FastAPI (app = FastAPI(...))
from app_config import (
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    MODEL_VERSION,
    PATH_TO_MODEL,
    PATH_TO_PREPROCESSOR,
)

from fastapi import FastAPI, HTTPException

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION
)

from lib.inference import run_inference
from lib.models import InputData, PredictionResult
from lib.utils import load_model, load_preprocessor

@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}

model = load_model(PATH_TO_MODEL)
preprocessor = load_preprocessor(PATH_TO_PREPROCESSOR)

@app.post("/predict", response_model=PredictionResult, status_code=201)
def predict(payload: InputData) -> dict:
    """Make predictions using the model."""
    try:
        # Convert the payload to a dictionary
        input_data = payload.dict()

        # Run inference (preprocess and predict)
        result = run_inference(input_data, preprocessor, model)

        # Return the prediction as a response
        return {"predicted_rings": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
