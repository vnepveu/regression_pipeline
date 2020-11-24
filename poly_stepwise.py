import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
from typing import Tuple

def polynomial_regression(dataset : pd.DataFrame,degree : float) -> Tuple[np.array, np.array, np.array, np.array, np.array] :

    """Split the data into a train and test set and return the associated
    polynomial features.
    :param dataset: loaded dataset.
    :param degree: the polynomial maximum degree to which the features will be developed.
    :return: training and test features and labels.
    """

    # Extract labels and features from data
    nb_columns = len(dataset.columns)
    y = dataset.iloc[:,nb_columns-1]
    x = dataset.iloc[:,0:nb_columns-1]


    # Transform features into their higher order terms. the final number of columns is given by C(n+d,d).
    # C() denotes the binomial operator.
    polynomial_features= PolynomialFeatures(degree=degree)

    x_poly = polynomial_features.fit_transform(x)

    return x_poly, y

def forward_regression(dataset : pd.DataFrame,
                       threshold_in: float,
                       verbose=True) -> Tuple[np.array, np.array, np.array, np.array, np.array]:

    """Perform a forward feature selection based on the p-value of each
    Ordinary least squares method step.
    :param dataset: loaded dataset.
    :param threshold in : value below which a feature is selected based
    on its p-value.
    :return: training, test features and labels.
    """

    # initial_list will contain selected feature names
    initial_list = []

    included = list(initial_list)
    nb_columns = len(dataset.columns)
    y = dataset.iloc[:,nb_columns-1]
    x = dataset.iloc[:,0:nb_columns-1]
    while True:
        changed=False
        excluded = list(set(x.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        # Select features minimum p-value
        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    # Create the new dataframe containing selected features
    models_fwd = pd.DataFrame(columns=included)

    for col in included:
        models_fwd[col] = x[col]

    return models_fwd, y

def backward_regression(dataset : pd.DataFrame,threshold_out: float,
                           verbose=True) -> Tuple[np.array, np.array, np.array, np.array, np.array]:

    """Perform a backward feature selection based on the p-value of each
    Ordinary least squares method step.
    :param dataset: loaded dataset.
    :param threshold in : value above which a feature is selected based
    on its p-value.
    :return: training, test features and labels.
    """

    nb_columns = len(dataset.columns)
    y = dataset.iloc[:,nb_columns-1]
    x = dataset.iloc[:,0:nb_columns-1]
    # included list will contain all feature names before starting to delete the selected ones.
    included=list(x.columns)

    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(x[included]))).fit()
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]

        # worst_pval will be null if pvalues is empty
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    # Create the new dataframe containing selected features
    models_fwd = pd.DataFrame(columns=included)

    for col in included:
        models_fwd[col] = x[col]
    
    return models_fwd, y