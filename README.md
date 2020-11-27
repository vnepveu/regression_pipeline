# Linear regression pipeline

## Table of Contents

1. [Requirements and Installation](#requirements_installation)
2. [Usage](#usage)
3. [Datasets](#datasets)
  1. [Boston Housing](#boston_housing)
  2. [Prostate Cancer](#prostate_cancer)
4. [Feature Selection](#feature_selection)
  1. [Principal Component Analysis - PCA](#PCA)
  2. [Correlation](#correlation)
5. [Regression Models](#models)
  1. [Linear Regression](#linear)
  2. [Lasso Regression](#lasso)
  3. [Ridge Regression](#ridge)
  4. [Elastic-Net Regression](#elasticnet)
  5. [Step-wise Forward Regression](#forward)
  6. [Step-wise Backward Regression](#backward)
  7. [Polynomial Regression](#polynomial)
6. [References](#references)

## Requirements and Installation <a name="requirements_installation"></a>

- python 3.8
- pip version ≥ 20.0

```
pip install -r requirements.txt
```

## Usage <a name="usage"></a>

In the the root of the repository, run the following command:
```
python ./pipeline_regression/main.py ./datasets/filename.csv model_name -f feature_selection -n n_splits
```

If `feature_selection` argument is not provided, no preprocessing will be done.
If `n_splits` argument is not provided, there will be 2 cross-validation steps.

For more details, run:
```
python ./pipeline_regression/main.py -h
```

## Datasets <a name="datasets"></a>

### Boston Housing <a name="boston_housing"></a>

There are 13 features to predict the price houses in Boston (`MEDV` feature).

Description:
  - `CRIM`: per capita crime rate by town.
  - `ZN`: proportion of residential land zoned for lots over 25,000 square feet.
  - `INDUS`: proportion of non-retail business acres per town.
  - `CHAS`: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
  - `NOX`: nitric oxides concentration (parts per 10 million).
  - `RM`: average number of rooms per dwelling.
  - `AGE`: proportion of owner-occupied units built prior to 1940.
  - `DIS`: weighted distances to five Boston employment centres.
  - `RAD`: index of accessibility to radial highways.
  - `TAX`: full-value property-tax rate per $10,000.
  - `PTRATIO`: pupil-teacher ratio by town.
  - `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of african americans by town.
  - `LSTAT`: % lower status of population.
  - `MEDV` (target): median value of owner-occupied homes in $1000's.


### Prostate Cancer <a name="prostate_cancer"></a>

The last column of the original dataset `train` has been removed.

There are 8 features to predict a log of PSA (Prostate Specific Antigen) value, a marker for Prostate cancer (`lpsa` feature).

Description:
  - `lcavol`: log of cancer volume.
  - `lweight`: log of prostate weight.
  - `age`: age of subject.
  - `lbph`: log of benign prostatic hyperplasia amount.
  - `svi`: seminal vesicle invasion.
  - `lcp`: log of capsular penetration.
  - `gleason`: Gleason score.
  - `pgg45`: percentage Gleason scores 4 or 5.
  - `lpsa` (target): log of PSA.

## Feature selection <a name="feature_selection"></a>

### Principal Component Analysis - PCA <a name="pca"></a>

Principal Component Analysis (PCA) is a dimensionality reduction technique that aims to reduce the number or features in the dataset. The technique is to combine the features in one matrix and extract the eigenvectors associated to the k highest eigenvalues in order to become the new features.

PCA when combined with linear regression is called Principal component regression (PCR) algorithm. The main issue we face with multi variate features data (data with considerable number of features) is the high variance of the model, which implies high instability of the prediction.

Consequently, the goal of reducing the dimension is to add more bias to the model, and therefore to penalize the accuracy but to guarantee less variance of the predictions.

The steps of the algorithm consist on creating the principal components from the existing features at first. Then we train the model on the components and transform the PCs back to the original features to make predictions on the actual dataset.

One of the main advantages of performing PCA before regression is to reduce the spatial and temporal complexity of the algorithm, and hence gain in performance. The second is to deal with overfitting especially on data with high colinearity.

The crucial decision to take before performing the PCA is to choose the number of output features. In our model, we choose to select the features depending on a desired level of variance to keep in our data.


### Correlation <a name="correlation"></a>

This technique takes in parameter the number 'k' of features to keep to train the model during next step.
The SelectKbest function using f_regression as score function, compute the correlation between each of the input feature and the label column.
Only the'k' inputs having the higher correlation factors are kept for the model training.


## Regression Models <a name="models"></a>

### Linear Regression <a name="linear"></a>

### Lasso Regression <a name="lasso"></a>

### Ridge Regression <a name="ridge"></a>

Ridge regression can be used for feature selection.
It is a regularization method that learns which features contribute best to the accuracy of the model while the model is being crated.
It introduces additional constraints into the cost function and drive the model toward lower complexity (fewer coefficients).

### Elastic-net Regression <a name="elasticnet"></a>

### Step-wise Forward Regression <a name="forward"></a>
forward selection + linear

### Step-wise Backward Regression <a name="backward"></a>
backward selection + linear

### Polynomial Regression <a name="polynomial"></a>
polynomial selection + linear <a name="introduction1"></a>

## References <a name="references"></a>

PCA
[] https://en.wikipedia.org/wiki/Principal_component_regression
https://iq.opengenus.org/principal-component-regression/

Prostate Cancer
[] https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data

Boston Housing
[] https://www.kaggle.com/altavish/boston-housing-dataset
