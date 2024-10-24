import numpy as np
import pandas as pd
from prefect import task
from preprocessing import preprocess_data
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error


@task(name="Make predictions")
def predict(df: pd.DataFrame, model: BaseEstimator) -> np.ndarray:
    """Predict Rings based on df data and model"""

    df = preprocess_data(df)

    X = df.drop(columns=["Rings"], errors="ignore")

    # Run predictions using the loaded model
    y_pred = model.predict(X)

    return y_pred


@task(name="Evaluate model")
def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error for two arrays"""
    return mean_squared_error(y_true, y_pred, squared=False)
