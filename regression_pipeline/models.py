import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression


def linear_regression() -> BaseEstimator:
    """Initialize a linear regression."""
    linear = LinearRegression()
    return linear


def ridge_regression() -> BaseEstimator:
    """Initialize a ridge regression."""
    ridge = Ridge()
    return ridge


def lasso_regression() -> BaseEstimator:
    """Initialize a lasso regression."""
    lasso = Lasso()
    return lasso


def elastic_net_regression() -> BaseEstimator:
    """Initialize a elastic-net regression."""
    elastic_net = ElasticNet()
    return elastic_net
