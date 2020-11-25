import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet


def ridge_regression(
    x_train: np.array, y_train: np.array, x_test: np.array
) -> BaseEstimator:
    """Ridge regression prediction.
    :param x_train: Training set
    :param y_train:Training labels
    :param x_test: Test set
    :return: Ridge model
    """
    ridge = Ridge()
    return ridge


def lasso_regression(
    x_train: np.array, y_train: np.array, x_test: np.array
) -> BaseEstimator:
    """Lasso regression prediction.
    :param x_train: Training set
    :param y_train:Training labels
    :param x_test: Test set
    :return: Lasso model
    """
    lasso = Lasso()
    return lasso


def elastic_net_regression(
    x_train: np.array, y_train: np.array, x_test: np.array
) -> BaseEstimator:
    """ElsaticNet regression prediction.
    :param x_train: Training set
    :param y_train:Training labels
    :param x_test: Test set
    :return: ElasticNet model
    """
    elastic_net = ElasticNet()
    return elastic_net
