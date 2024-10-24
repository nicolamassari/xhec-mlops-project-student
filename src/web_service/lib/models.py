# Pydantic models for the web service

from pydantic import BaseModel


class InputData(BaseModel):
    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float
    sex: str


class PredictionOut(BaseModel):
    rings: float
