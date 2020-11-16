import os

import pandas as pd
from pandas.api.types import is_integer_dtype, is_float_dtype


def load_dataset(filename: str) -> pd.DataFrame:
    """Load a dataset from its filename.

    :param filename: path to the dataset's `.csv` file.
    :return: the loaded dataset in a dataframe.
    """
    dataset_df = pd.read_csv(filename)
    return dataset_df


def _prepare_float(
    column_series: pd.Series, mean: bool = True, standardize: bool = True
) -> pd.Series:
    """
    Fill missing values of a float column with its mean or median,
    and standardize it.

    :param column_series: column to process.
    :param mean: whether to fill missing values with the mean or the median.
    :param standardize: whether to standardize or normalize the column.
    :return: the processed column.
    """
    # Fill missing values with the mean or the median of the column
    filling_value = column_series.mean() if mean else column_series.median()
    column_series.fillna(filling_value, inplace=True)

    # Standardize the column
    a = column_series.mean() if standardize else column_series.min()
    b = column_series.std() if standardize else column_series.max() - a
    column_series = column_series.apply(lambda x: (x - a) / b)

    return column_series


def _prepare_int(column_series: pd.Series, mean: bool = True) -> pd.Series:
    """Fill missing values of an integer column with its mean or median.

    :param column_series: column to process.
    :param mean: whether to fill missing values with the mean or the median.
    :return: the processed column.
    """
    # Fill missing values with the mean or the median of the column
    filling_value = column_series.mean() if mean else column_series.median()
    column_series.fillna(int(filling_value), inplace=True)

    return column_series


def _prepare_bool(column_series: pd.Series) -> pd.Series:
    """Fill missing values of a bolean column with the most frequent value.

    :param column_series: column to process.
    :return: the processed column.
    """
    filling_value = column_series.mode()
    column_series.fillna(int(filling_value), inplace=True)

    return column_series


def prepare_data(dataset_df: pd.DataFrame):
    """Fill missing values and standardize float columns.

    :param dataset_df: dataset to process.
    """
    for column_name, column_series in dataset_df.iteritems():
        if is_integer_dtype(column_series):
            if set(column_series.unique()) == {0, 1}:
                dataset_df[column_name] = _prepare_bool(column_series)
            else:
                dataset_df[column_name] = _prepare_int(column_series)
        elif is_float_dtype(column_series):
            dataset_df[column_name] = _prepare_float(column_series)

        # Raise an error is the column type is not boolean, integer or float
        else:
            raise TypeError(f"Unrecognized type, column: {column_name}")
