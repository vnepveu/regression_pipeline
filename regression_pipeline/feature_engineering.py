import numpy as np
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_regression,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import pandas as pd

# Disable pandas' warning when copying a column
pd.set_option("mode.chained_assignment", None)


def select_ridge_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Select the relevant features using Ridge Regression method.

    :param dataset_df: a loaded dataset.
    :return: the selected features.
    """
    # Splitting label column from input column
    label_column = dataset_df.iloc[:, -1]
    X = dataset_df.iloc[:, :-1]

    # Using Ridge regression to select the relevant features
    selection = SelectFromModel(Ridge())
    selection.fit(X, label_column)
    selected_features = dataset_df.columns[(selection.get_support())]

    new_dataset_df = dataset_df[selected_features]
    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df


def select_correlation_features(
    dataset_df: pd.DataFrame, n_features: int = 5
) -> pd.DataFrame:
    """Select the relevant features by looking at correlation.

    :param dataset_df: a loaded dataset.
    :param n_features: number of top features to select.
    :return: the selected features.
    """
    # Splitting label column from input column
    label_column = dataset_df.iloc[:, -1]
    X = dataset_df.iloc[:, :-1]

    # Using `f_regression` to select the relevant features
    selection = SelectKBest(score_func=f_regression, k=n_features)
    selection.fit(X, label_column)
    selected_features = X.columns[(selection.get_support())]

    new_dataset_df = dataset_df[selected_features]
    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df


def select_pca_features(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load a dataset and apply principal component analysis, in order to reduce
    dimensionality.

    :param dataset_df: a loaded dataset.
    :return: transformed dataframe with reduced features and labels.
    """
    nb_columns = len(dataset_df.columns)

    # Depending on the number of features, the method need more or less variance
    # (expressed in percentage) in order to return good number of components
    if nb_columns > 10:
        pca = PCA(0.9999)
    else:
        pca = PCA(0.999)

    # Label column of dataset is located at the last position
    label_column = dataset_df.iloc[:, nb_columns - 1]
    X = dataset_df.iloc[:, : (nb_columns - 1)]
    selected_features = pca.fit_transform(X)

    # Concatenating transformed columns into one dataframe
    new_dataset_df = pd.DataFrame(
        data=selected_features,
        columns=[f"PC{i}" for i in range(np.shape(selected_features)[1])],
    )

    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df


def select_polynomial_features(
    dataset_df: pd.DataFrame, degree: float = 3
) -> pd.DataFrame:
    """
    Split the data into a train and test set and return the associated
    polynomial features.

    :param dataset_df: a loaded dataset.
    :param degree: maximum polynomial degree to develop features.
    :return: transformed dataframe with reduced features and labels.
    """
    # Extract labels and features from data
    nb_columns = len(dataset_df.columns)
    label_column = dataset_df.iloc[:, nb_columns - 1]
    X = dataset_df.iloc[:, : nb_columns - 1]

    # Transform features into their higher order terms, the final number of
    # columns is given by C(n+d,d). C() denotes the binomial operator
    polynomial_features = PolynomialFeatures(degree=degree)
    selected_features = polynomial_features.fit_transform(X)

    # Concatenating transformed columns into one dataframe
    new_dataset_df = pd.DataFrame(
        data=selected_features,
        columns=[f"X{i}" for i in range(np.shape(selected_features)[1])],
    )

    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df


def select_forward_features(
    dataset_df: pd.DataFrame, threshold_in: float = 0.05, verbose=False
) -> pd.DataFrame:
    """
    Perform a forward feature selection based on the p-value of each ordinary
    least squares method step.

    :param dataset_df: a loaded dataset.
    :param threshold_in: select features below (based on its p-value).
    :return: transformed dataframe with reduced features and labels.
    """
    # initial_list will contain selected feature names
    initial_list = []

    included = list(initial_list)
    nb_columns = len(dataset_df.columns)
    label_column = dataset_df.iloc[:, nb_columns - 1]
    X = dataset_df.iloc[:, : nb_columns - 1]
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(
                label_column,
                sm.add_constant(pd.DataFrame(X[included + [new_column]])),
            ).fit()
            new_pval[new_column] = model.pvalues[new_column]
        # Select features minimum p-value
        best_pval = new_pval.min()

        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print(f"Add {best_feature:10} - p-value {best_pval:.2E}")

        if not changed:
            break

    # Create the new dataframe containing selected features
    new_dataset_df = pd.DataFrame(columns=included)
    for column_name in included:
        new_dataset_df[column_name] = X[column_name]

    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df


def select_backward_features(
    dataset_df: pd.DataFrame, threshold_out: float = 0.01, verbose=False
) -> pd.DataFrame:
    """
    Perform a backward feature selection based on the p-value of each ordinary
    least squares method step.

    :param dataset_df: a loaded dataset.
    :param threshold_out: select features above (based on its p-value).
    :return: transformed dataframe with reduced features and labels.
    """
    nb_columns = len(dataset_df.columns)
    label_column = dataset_df.iloc[:, nb_columns - 1]
    X = dataset_df.iloc[:, : nb_columns - 1]
    # Included list will contain all feature names before starting to delete
    # the selected ones
    included = list(X.columns)

    while True:
        changed = False
        model = sm.OLS(
            label_column, sm.add_constant(pd.DataFrame(X[included]))
        ).fit()
        # Use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]

        # worst_pval will be null if pvalues is emptlabel_column
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print(f"Drop {worst_feature:10} - p-value: {worst_pval:.2E}")
        if not changed:
            break

    # Create the new dataframe containing selected features
    new_dataset_df = pd.DataFrame(columns=included)
    for column_name in included:
        new_dataset_df[column_name] = X[column_name]

    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df
