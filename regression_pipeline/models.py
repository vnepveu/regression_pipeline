from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression


def linear_regression() -> BaseEstimator:
    """Initialize a linear regression.
    
    :author: Victor Nepveu
    """
    linear = LinearRegression()
    return linear


def ridge_regression(alpha=1) -> BaseEstimator:
    """Initialize a ridge regression.
    
    :author: Antoine Lerosey
    :param alpha: regularization strenght
    """
    ridge = Ridge(alpha)
    return ridge


def lasso_regression(alpha=0.1) -> BaseEstimator:
    """Initialize a lasso regression.

    :author: Antoine Lerosey
    :param alpha: constant multiplying the L1 penalty term
    """
    lasso = Lasso(alpha)
    return lasso


def elastic_net_regression(alpha=0.1, l1_ratio=0.3) -> BaseEstimator:
    """Initialize a elastic-net regression.

    :author: Robin Courant
    :param alpha: constant multiplying L1 and L2 penalty term
    :param l1_ratio: the mixing parameter between L1 and L2 term.
    0<=l1_ratio<=1. If l1_ration=0 this is an L2 penalty
    """
    elastic_net = ElasticNet(alpha, l1_ratio)
    return elastic_net
