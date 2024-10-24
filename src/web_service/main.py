# Code with FastAPI (app = FastAPI(...))
from app_config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, PATH_TO_MODEL
from fastapi import FastAPI, HTTPException

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

from lib.inference import run_inference
from lib.models import InputData, PredictionOut
from lib.preprocessing import preprocess_data
from lib.utils import load_model


@app.get("/")
def home() -> dict:
    return {"health_check": "App up and running!"}


model = load_model(PATH_TO_MODEL)


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData) -> dict:
    """Make predictions using the model."""
    try:
        # Run inference (preprocess and predict)
        result = run_inference([payload], preprocess_data, model)

        # Return the prediction as a response
        return {"predicted_rings": result}

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
