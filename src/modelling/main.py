import argparse
import pickle
from pathlib import Path

import pandas as pd
from config import NUMERICAL_FEATURES, OUTLIER_CONDITIONS, SEX_MAPPING, TARGET
from preprocessing import preprocess_data
from training import train_rf


def main(trainset_path: Path) -> None:
    """Train a model using the data at the given path and save the model (pickle)."""
    # Read data
    df = pd.read_csv(trainset_path)

    # Preprocess data
    X_train, y_train = preprocess_data(
        df, "Sex", SEX_MAPPING, OUTLIER_CONDITIONS, NUMERICAL_FEATURES, TARGET
    )

    # Train model
    model = train_rf(X_train, y_train)

    # Pickle model
    with open("../src/web_service/local_objects/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model using the data at the given path."
    )
    parser.add_argument("trainset_path", type=str, help="Path to the training set")
    args = parser.parse_args()
    main(args.trainset_path)
