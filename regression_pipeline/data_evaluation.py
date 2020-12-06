from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


def get_predictions_cv(
    X: np.array, y_true: np.array, model: BaseEstimator, n_splits: int
) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
    """Perform a cross-test with the provided model.

    :param X: design matrix (n_features, n_samples).
    :param y_true: target vector (1, n_samples).
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


def get_score_cv(Y_pred, Y_test, markdown=True):
    """Compute median and mean r2 and MSE over cross-validation predictions.

    :param Y_pred: test predictions (n_splits, n_samples, ).
    :param Y_test: test target (n_splits, n_samples, ).
    :rerturn: median and mean of MSE and r2 scores.
    """
    # Compute scores for each cross-validation step
    scores = defaultdict(list)
    for y_pred, y_test in zip(Y_pred, Y_test):
        scores["MSE"].append(mean_squared_error(y_pred, y_test))
        scores["r2"].append(r2_score(y_pred, y_test))

    # Store these score in a table
    scores_df = pd.DataFrame.from_dict(
        data=scores,
        columns=[f"CV {k + 1}" for k in range(len(Y_pred))],
        orient="index",
    )
    # Compute median, mean and standard-diviation of these scores
    scores_df.loc[:, "median"] = pd.Series(
        [scores_df.loc["MSE"].median(), scores_df.loc["r2"].median()],
        index=scores_df.index,
    )
    scores_df.loc[:, "mean"] = pd.Series(
        [scores_df.loc["MSE"].mean(), scores_df.loc["r2"].mean()],
        index=scores_df.index,
    )
    scores_df.loc[:, "std"] = pd.Series(
        [scores_df.loc["MSE"].std(), scores_df.loc["r2"].std()],
        index=scores_df.index,
    )
    if markdown:
        return scores_df.round(2).transpose().to_markdown()

    return scores_df.round(2).transpose()
