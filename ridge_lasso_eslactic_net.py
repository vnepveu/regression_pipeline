import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score


def ridge_regression(
    x_train: np.array, y_train: np.array, x_test: np.array
) -> np.array:
    """Ridge regression prediction.
    :param x_train: Training set
    :param y_train:Training labels
    :param x_test: Test set
    :return: prediction
    """
    ridge = Ridge()
    ridge.fit(x_train, y_train)
    return ridge.predict(x_test)


def lasso_regression(
    x_train: np.array, y_train: np.array, x_test: np.array
) -> np.array:
    """Lasso regression prediction.
    :param x_train: Training set
    :param y_train:Training labels
    :param x_test: Test set
    :return: prediction
    """
    lasso = Lasso()
    lasso.fit(x_train, y_train)
    return lasso.predict(x_test)


def elastic_net_regression(
    x_train: np.array, y_train: np.array, x_test: np.array
) -> np.array:
    """ElsaticNet regression prediction.
    :param x_train: Training set
    :param y_train:Training labels
    :param x_test: Test set
    :return: prediction
    """
    elastic_net = ElasticNet()
    elastic_net.fit(x_train, y_train)
    return elastic_net.predict(x_test)


def score(predictions: np.array, y_test: np.array) -> [float, float]:

    """Mean squared error and r2 score of the model prediction.
    :param predictions : Predictions from the model.
    :param y_test : Test labels
    :return: regression scores.
    """
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return rmse, r2
