from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator

from modelling.preprocessing import encode_categorical_cols

from .models import InputData


def run_inference(
    input_data: List[InputData], encoder: BaseEstimator, model: BaseEstimator
) -> np.ndarray:
    """Run inference on a list of input

    Args:
        input_data (List[InputData]): A li
        encoder (BaseEstimator): The fitted en
        model (BaseEstimator): The fitted model object.

    Returns:
        np.ndarray: The predicted number of rings for each input data point.

    Example InputData:
        {'Sex': 'M', 'Length': 0.455, 'Diameter': 0.365, 'Height': 0.095,
        'Whole weight': 0.514, 'Shucked weight': 0.2245,
        'Viscera weight': 0.101, 'Shell weight': 0.15}
    """
    logger.info(f"Running inference on:\n{input_data}")

    # Convert input_data list into a DataFrame
    df = pd.DataFrame([x.dict() for x in input_data])

    # Encode the categorical columns (e.g., "Sex")
    df = encode_categorical_cols(df, encoder)

    # Separate the features used for predictions
    X = df.drop(columns=["Rings"], errors="ignore")

    # Run predictions using the loaded model
    y_pred = model.predict(X)
    logger.info(f"Predicted number of rings:\n{y_pred}")
    return y_pred
