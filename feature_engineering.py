import pandas as pd
import numpy as np
#import sys
#sys.path.insert(1,r"C:\Users\Yakoubi\regression_pipeline")
from data_preparation import *
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectFromModel

def feature_selection_ridge(dataset:pd.DataFrame):
    """
    Function to select the relevant features to build a model
    """
    data_work = load_dataset(dataset)
    prepare_data(data_work) # Perform data preparation before coming up with PCA
    nb_columns = len(data_work.columns)
    
    #splitting label column from input column
    label_column=data_work.iloc[:,-1] 
    data_work=data_work.iloc[:,:-1]

    #Using Ridge regression to select the relevant
    sel_=SelectFromModel(Ridge())
    sel_.fit(data_work,label_column)
    selected_features = data_work.columns[(sel_.get_support())]
    return selected_features
