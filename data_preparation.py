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
    column_series: pd.Series,
    mean: bool = True,
    rescale: bool = True,
    standardize: bool = True,
) -> pd.Series:
    """
    Fill missing values of a float column with its mean or median,
    and standardize it.

    :param column_series: column to process.
    :param mean: whether to fill missing values with the mean or the median.
    :param rescale: whether to rescale the column (standardize or normalize).
    :param standardize: whether to apply standardization or normalization.
    :return: the processed column.
    """
    # Fill missing values with the mean or the median of the column
    filling_value = column_series.mean() if mean else column_series.median()
    column_series.fillna(filling_value, inplace=True)

    if rescale:
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


def prepare_data(
    dataset_df: pd.DataFrame,
    fill_bool: bool = True,
    fill_int: bool = True,
    fill_float: bool = True,
    mean_int: bool = True,
    mean_float: bool = True,
    rescale_float: bool = True,
    standardize_float: bool = True,
):
    """Fill missing values and standardize float columns.

    :param dataset_df: dataset to process.
    :param fill_bool: whether to fill missing boolean values.
    :param fill_int: whether to fill missing integer values.
    :param fill_float: whether to fill missing float values.
    :param mean_int: whether to use mean or the median for missing integers.
    :param mean_float: whether to use mean or the median for missing floats.
    :param rescale_float: whether to rescale floats (standardize or normalize).
    :param standardize_float: whether to apply standardization or normalization.
    """
    for column_name, column_series in dataset_df.iteritems():
        if is_integer_dtype(column_series):
            if set(column_series.unique()) == {0, 1}:
                if fill_bool:
                    dataset_df[column_name] = _prepare_bool(column_series)
            elif fill_int:
                dataset_df[column_name] = _prepare_int(column_series, mean_int)
        elif is_float_dtype(column_series):
            if fill_float:
                dataset_df[column_name] = _prepare_float(
                    column_series, mean_float, rescale_float, standardize_float
                )
        # Raise an error is the column's type is not boolean, integer or float
        else:
            raise TypeError(f"Unrecognized type, column: {column_name}")
