from typing import Tuple

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold


def get_predictions_cv(
    X: np.array, y_true: np.array, model: BaseEstimator, n_splits: int
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """Perform a cross-test with the provided model.

    :param X: design matrix (n_features, n_samples).
    :param X: target vector (1, n_samples).
    :param model: model used to make predictions.
    :param n_splits: number of folds (must be at leat 2).
    :return:
        - `X_train`: training sets (n_splits, n_samples, n_features).
        - `X_test`: test sets (n_splits, n_samples, n_features).
        - `Y_train`: training targets (n_splits, n_samples, ).
        - `Y_test`: test target (n_splits, n_samples, ).
        - `Y_pred`: test predictions (n_splits, n_samples, ).
    """
    X_train, X_test = [], []
    Y_train, Y_test, Y_pred = [], [], []

    kf = KFold(n_splits=n_splits, shuffle=True)
    for train_index, test_index in kf.split(X):
        # Split a new train and test sets for each round
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y_true[train_index], y_true[test_index]

        # Clone the unfitted model in order retrain a model for each round
        k_model = clone(model)
        k_model.fit(x_train, y_train)
        # Compute predictions on test set
        y_pred = k_model.predict(x_test)

        # Store train, test and prediction sets
        X_train.append(x_train)
        X_test.append(x_test)
        Y_train.append(y_train)
        Y_test.append(y_test)
        Y_pred.append(y_pred)

    return (
        X_train,
        X_test,
        Y_train,
        Y_test,
        Y_pred,
    )
