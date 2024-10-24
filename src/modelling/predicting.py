import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from modelling.preprocessing import preprocessing


def predict(df: pd.DataFrame, model: BaseEstimator) -> np.ndarray:
    """Predict Rings based on df data and model"""

    df = preprocessing(df)

    X = df.drop(columns=["Rings"], errors="ignore")

    # Run predictions using the loaded model
    y_pred = model.predict(X)

    return y_pred
