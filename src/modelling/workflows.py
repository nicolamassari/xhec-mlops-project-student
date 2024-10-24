import os

import pandas as pd
from config import NUMERICAL_FEATURES, OUTLIER_CONDITIONS, SEX_MAPPING
from loguru import logger
from predicting import evaluate_model, predict
from prefect import flow
from preprocessing import preprocess_data
from training import train_rf
from utils import save_pickle


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
