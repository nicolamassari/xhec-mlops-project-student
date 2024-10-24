from typing import List

import pandas as pd
from prefect import flow, task


@task
def label_encode_column(
    df: pd.DataFrame, categorical_feature: str, encoding_map: dict
) -> pd.DataFrame:
    """
    Apply label encoding to a categorical column in the dataframe.

    Parameters:
    - df (pd.DataFrame): The dataframe to modify.
    - categorical_feature (str): The name of the categorical column to encode.
    - encoding_map (dict): A mapping of categorical values to numeric codes.

    Returns:
    - pd.DataFrame: The dataframe with the encoded categorical column.
    """
    df[categorical_feature] = df[categorical_feature].replace(encoding_map)
    return df


@task
def remove_outliers(df: pd.DataFrame, conditions: list) -> pd.DataFrame:
    """
    Remove outliers from the dataframe based on a list of conditions.

    Parameters:
    - df (pd.DataFrame): The dataframe to filter.
    - conditions (list): A list of conditions in the form of tuples
                         (column, operator, value) for outlier removal.

    Returns:
    - pd.DataFrame: The dataframe with outliers removed.
    """
    for column, operator, value in conditions:
        if operator == ">":
            df = df[df[column] <= value]
        elif operator == "<":
            df = df[df[column] >= value]
        elif operator == ">=":
            df = df[df[column] < value]
        elif operator == "<=":
            df = df[df[column] > value]
    return df


@task
def extract_x_y(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    with_target: bool = True,
) -> dict:
    """
    Extract features (X) and target (y) from the dataframe.

    Parameters:
    - df: The input dataframe containing the dataset.
    - categorical_cols: A list of categorical column names.
    - numerical_cols: A list of numerical column names.
    - with_target: wether the target should be included in the output or not.

    Returns:
    - A tuple containing:
        - X: A dataframe with selected feature columns.
        - y: The target variable values.
    """
    features = numerical_cols + categorical_cols
    X = df[features]

    y = None

    if with_target:
        y = df["Rings"].values

    return X, y


@flow(name="Preprocess data")
def preprocess_data(
    df: pd.DataFrame,
    categorical_feature: str,
    encoding_map: dict,
    outlier_conditions: list,
    numerical_cols: List[str],
    with_target: bool = True,
) -> tuple:
    """
    Preprocessing pipeline: Encode categorical features, remove outliers,
    and extract features and target.

    Parameters:
    - df: The input dataframe containing the dataset.
    - categorical_feature: The name of the categorical column to encode.
    - encoding_map: A mapping of categorical values to numeric codes.
    - outlier_conditions: A list of conditions for outlier removal.
    - numerical_cols: A list of numerical columns to include in the feature set.
    - with_target: Boolean defining if the target should be included in the output.

    Returns:
    - A tuple containing:
        - X: Preprocessed features.
        - y: Preprocessed target variable.
    """
    df = label_encode_column(df, categorical_feature, encoding_map)
    df = remove_outliers(df, outlier_conditions)

    categorical_cols = [categorical_feature]
    X, y = extract_x_y(df, categorical_cols, numerical_cols, with_target)

    return X, y
