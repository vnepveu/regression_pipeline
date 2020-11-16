import pandas as pd
import numpy as np
#import sys
#sys.path.insert(1,r"C:\Users\Yakoubi\regression_pipeline")
from data_preparation import *
from sklearn.decomposition import PCA


def reduce_dimensionality(dataset: pd.DataFrame):
    # Function to load dataset and apply principal component analysis in order to reduce dimensionality. Return the dimension reduced features and the labels in one dataframe.
    data_work = load_dataset(dataset)
    prepare_data(data_work) # Perform data preparation before coming up with PCA
    nb_columns = len(data_work.columns)
    if(nb_columns > 10): # Depending on the number of features, the PCA algorithm will need more or less variance (expressed in percentage) in order to return adequate number of components.
        pca = PCA(.9999)
    else:
        pca = PCA(.999)
    label_column = data_work.iloc[:,nb_columns-1] # label column of dataset is located at the last position.
    data_work = data_work.iloc[:,0:nb_columns-1]
    data_transformed = pca.fit_transform(data_work)
    data_transformed=pd.DataFrame(data=data_transformed,columns=[f'principal components {i}' for i in range (np.shape(data_transformed)[1])]) # Concatenating transformed columns into one dataframe.
    data_transformed['label'] = label_column # appending column label to the transformed dataframe at the last position.
    return data_transformed