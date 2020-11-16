import pandas as pd
import numpy as np
from data_preparation import *
from sklearn.decomposition import PCA


def select_feature_pca(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Function to load dataset and apply principal component analysis,
    in order to reduce dimensionality.

    :param dataset: dataset to process.
    :return: Transformed dataframe with reduced features and labels.
    """
    data_work = load_dataset(dataset)
    # Perform data preparation before coming up with PCA
    prepare_data(data_work)
    nb_columns = len(data_work.columns)

    # Depending on the number of features, the PCA algorithm will need more or less variance (expressed in percentage) in order
    # to return adequate number of components.
    if(nb_columns > 10):
        pca = PCA(.9999)
    else:
        pca = PCA(.999)

    # Label column of dataset is located at the last position.
    label_column = data_work.iloc[:,nb_columns-1]
    data_work = data_work.iloc[:,0:nb_columns-1]
    data_transformed = pca.fit_transform(data_work)

    # Concatenating transformed columns into one dataframe.
    data_transformed=pd.DataFrame(data=data_transformed,columns=[f'principal components {i}' for i in range (np.shape(data_transformed)[1])])

    # Appending column label to the transformed dataframe at the last position.
    data_transformed['label'] = label_column 
    
    return data_transformed