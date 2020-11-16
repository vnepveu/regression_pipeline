import numpy as np

import pandas as pd

from sklearn.feature_selection import SelectFromModel, SelectKBest,f_regression
from sklearn.linear_model import Ridge



from data_preparation import *

def select_features_ridge(filename:str):
    """Select the relevant features using Ridge Regression method.
    :param filename: path to the dataset's '.csv' file
    :return: the selected features in an Index type

    """
    data_work = load_dataset(dataset)
    # Perform data preparation before selecting features
    prepare_data(data_work)
    nb_columns = len(data_work.columns)
    
    # Splitting label column from input column
    label_column=data_work.iloc[:,-1] 
    data_work=data_work.iloc[:,:-1]

    # Using Ridge regression to select the relevant features
    selection= SelectFromModel(Ridge())
    selection.fit(data_work,label_column)
    selected_features = data_work.columns[(selection.get_support())]
    return selected_features

def select_features_correlation(filename:str,number_of_feature_to_keep:int):
    """Select the relevant features by looking at correlation
    :param filename: path to the datset's '.csv' file
    :param number_of_features_to_keep: number of top features to select
    :return: the selected feature in an Index type
    """

    data_work = load_dataset(dataset)
    # Perform data preparation before selecting features
    prepare_data(data_work)
    nb_columns = len(data_work.columns)

    # Splitting label column from input column
    label_column = data_work.iloc[:,-1] 
    data_work = data_work.iloc[:,:-1]

    # Using f_regression to select the relevant features
    selection = SelectKBest(score_func=f_regression, k=number_of_feature_to_keep)
    selection.fit(data_work,label_column)
    selected_features = data_work.columns[(selection.get_support())]
    return selected_features
    

