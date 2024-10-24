from prefect import task, flow
from pathlib import Path
import pandas as pd
import os
import pickle
from loguru import logger

from preprocessing import preprocess_data
from training import train

@task
def read_data(file_path: str) -> pd.DataFrame:
    """Read dataset from the provided file path."""
    return pd.read_csv(file_path)

@task
def preprocess_data_task(
    df: pd.DataFrame,
    categorical_feature: str,
    encoding_map: dict,
    outlier_conditions: list,
    numerical_cols: list,
    target: str,
):
    """Preprocess the data using predefined preprocessing functions."""
    return preprocess_data(
        df, categorical_feature, encoding_map, outlier_conditions, numerical_cols, target
    )

@task
def train_model_task(X_train, y_train):
    """Train the model using the provided training data."""
    return train(X_train, y_train)

@task
def save_model_task(model, model_filepath: str):
    """Save the trained model to a file in pickle format."""
    os.makedirs(os.path.dirname(model_filepath), exist_ok=True)
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)

@flow(name="Training Workflow")
def train_model_workflow(
    train_filepath: str,
    categorical_feature: str,
    encoding_map: dict,
    outlier_conditions: list,
    numerical_cols: list,
    target: str,
    model_filepath: str,
):
    """Train a model using the data at the given path and save the model (pickle)."""
    logger.info("Reading training data...")
    df = read_data(train_filepath).result()

    logger.info("Preprocessing data...")
    X_train, y_train = preprocess_data_task(
        df,
        categorical_feature=categorical_feature,
        encoding_map=encoding_map,
        outlier_conditions=outlier_conditions,
        numerical_cols=numerical_cols,
        target=target,
    ).result()

    logger.info("Training model...")
    model = train_model_task(X_train, y_train).result()

    logger.info("Saving model...")
    save_model_task(model, model_filepath)
