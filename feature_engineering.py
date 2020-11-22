import numpy as np
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    f_regression,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import pandas as pd

# Disable pandas' warning when copying a column
pd.set_option("mode.chained_assignment", None)


def select_features_ridge(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Select the relevant features using Ridge Regression method.

    :param dataset_df: a loaded dataset.
    :return: the selected features.
    """
    # Splitting label column from input column
    label_column = dataset_df.iloc[:, -1]
    dataset_df = dataset_df.iloc[:, :-1]

    # Using Ridge regression to select the relevant features
    selection = SelectFromModel(Ridge())
    selection.fit(dataset_df, label_column)
    selected_features = dataset_df.columns[(selection.get_support())]

    new_dataset_df = dataset_df[selected_features]
    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df


def select_features_correlation(
    dataset_df: pd.DataFrame, n_features: int = 5
) -> pd.DataFrame:
    """Select the relevant features by looking at correlation.

    :param dataset_df: a loaded dataset.
    :param n_features: number of top features to select.
    :return: the selected features.
    """
    # Splitting label column from input column
    label_column = dataset_df.iloc[:, -1]
    dataset_df = dataset_df.iloc[:, :-1]

    # Using `f_regression` to select the relevant features
    selection = SelectKBest(score_func=f_regression, k=n_features)
    selection.fit(dataset_df, label_column)
    selected_features = dataset_df.columns[(selection.get_support())]

    new_dataset_df = dataset_df[selected_features]
    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df


def select_features_pca(dataset_df: pd.DataFrame) -> pd.DataFrame:
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
    dataset_df = dataset_df.iloc[:, : (nb_columns - 1)]
    selected_features = pca.fit_transform(dataset_df)

    # Concatenating transformed columns into one dataframe
    new_dataset_df = pd.DataFrame(
        data=selected_features,
        columns=[f"PC{i}" for i in range(np.shape(selected_features)[1])],
    )

    # Adding the label column to the transformed dataframe
    new_dataset_df["label"] = label_column

    return new_dataset_df
