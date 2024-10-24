from sklearn.ensemble import RandomForestRegressor


def train(X_train, y_train):
    """
    Trains a RandomForestRegressor model using the provided training data.

    Parameters:
    -----------
    X_train : array-like or pandas DataFrame
        The input features used for training the model.

    y_train : array-like or pandas Series
        The target values (labels) used for training the model.

    Returns:
    --------
    RandomForestRegressor
        A fitted RandomForestRegressor model with the specified hyperparameters.
    """

    regressor = RandomForestRegressor(
        bootstrap=False,
        max_depth=20,
        max_features="sqrt",
        min_samples_leaf=4,
        min_samples_split=5,
        n_estimators=150,
    )

    regressor.fit(X_train, y_train)

    return regressor
