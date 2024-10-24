from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator

from lib.models import InputData
from ...modelling.preprocessing import preprocess_data
from ...modelling.config import SEX_MAPPING, OUTLIER_CONDITIONS, NUMERICAL_FEATURES


def run_inference(input_data: List[InputData], model: BaseEstimator) -> np.ndarray:
    """Run inference on a list of input data.

    Args:
        payload (dict): the data point to run inference on.
        model (BaseEstimator): the fitted model object.

    Returns:
        np.ndarray: the predicted number of rings.
    """
    logger.info(f"Running inference on:\n{input_data}")
    df = pd.DataFrame([x.dict() for x in input_data])

    # Preprocess data
    X = preprocess_data(
        df, "Sex", SEX_MAPPING, OUTLIER_CONDITIONS, NUMERICAL_FEATURES, False
    )

    y = model.predict(X)
    logger.info(f"Predicted trip durations:\n{y}")
    return y
