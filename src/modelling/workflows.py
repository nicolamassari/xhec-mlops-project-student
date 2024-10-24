import os
from typing import Optional

import numpy as np
import pandas as pd
from config import NUMERICAL_FEATURES, OUTLIER_CONDITIONS, SEX_MAPPING
from loguru import logger
from predicting import evaluate_model, predict
from prefect import flow
from preprocessing import preprocess_data
from sklearn.ensemble import RandomForestRegressor
from training import train_rf
from utils import load_pickle, save_pickle


@flow(name="Training Workflow")
def train_model_workflow(
    train_filepath: str,
    test_filepath: str,
    model_filepath: str,
):
    """Train a model using the data at the given path and save the model (pickle)."""
    logger.info("Reading training data...")
    train = pd.read_csv(train_filepath)
    test = pd.read_csv(test_filepath)

    logger.info("Preprocessing data...")
    X_train, y_train = preprocess_data(
        train,
        "Sex",
        SEX_MAPPING,
        OUTLIER_CONDITIONS,
        NUMERICAL_FEATURES,
        True,
    )

    X_test, y_test = preprocess_data(
        test,
        "Sex",
        SEX_MAPPING,
        OUTLIER_CONDITIONS,
        NUMERICAL_FEATURES,
        True,
    )

    logger.info("Training model...")
    model = train_rf(X_train, y_train)

    logger.info("Making predictions and evaluating...")
    y_pred = predict(X_test, model)
    rmse = evaluate_model(y_test, y_pred)

    if model_filepath is not None:
        logger.info(f"Saving artifacts to {model_filepath}...")
        save_pickle(os.path.join(model_filepath, "model.pkl"), model)

    return {"model": model, "rmse": rmse}


@flow(name="Batch predict", retries=1, retry_delay_seconds=30)
def batch_predict_workflow(
    input_filepath: str,
    model: Optional[RandomForestRegressor] = None,
    model_filepath: Optional[str] = None,
) -> np.ndarray:
    """Make predictions on a new dataset"""
    df = pd.read_csv(input_filepath)

    if model is None:
        model = load_pickle(os.path.join(model_filepath, "model.pkl"))

    X, _ = preprocess_data(
        df, "Sex", SEX_MAPPING, OUTLIER_CONDITIONS, NUMERICAL_FEATURES, False
    )
    y_pred = predict(X, model)
    return y_pred
