# Linear regression pipeline

## Table of Contents

1. [Requirements and Installation](#requirements_installation)
1. [Usage](#usage)
1. [Datasets](#datasets)
   1. [Boston Housing](#boston_housing)
   1. [Prostate Cancer](#prostate_cancer)
1. [Feature Selection](#feature_selection)
   1. [Principal Component Analysis - PCA](#PCA)
   1. [Correlation](#correlation)
1. [Regression Models](#models)
   1. [Linear Regression](#linear)
   1. [Lasso Regression](#lasso)
   1. [Ridge Regression](#ridge)
   1. [Elastic-Net Regression](#elasticnet)
   1. [Step-wise Forward Regression](#forward)
   1. [Step-wise Backward Regression](#backward)
   1. [Polynomial Regression](#polynomial)
1. [References](#references)

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

- If `feature_selection` argument is not provided, no preprocessing will be done.
- If `n_splits` argument is not provided, there will be 2 cross-validation steps.
- If `model_name` is set to `all`, then all models are run. Redirecting the 
result with `>` can then be used to generate pretty markdown tables of the 
results.
For instance :
```
python ./pipeline_regression/main.py ./datasets/boston_housing.csv all -f pca -n 5 > boston_results.md
```
will generate a file called `boston_results.md` that is valid markdown.

- For more details, run:
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

### Feature selection <a name="feature_selection"></a>

## Principal Component Analysis - PCA <a name="pca"></a>

Principal Component Analysis (PCA) is a dimensionality reduction technique that aims to reduce the number or features in the dataset. The technique is to combine the features in one matrix and extract the eigenvectors associated to the k highest eigenvalues in order to become the new features.

PCA when combined with linear regression is called Principal component regression (PCR) algorithm. The main issue we face with multi variate features data (data with considerable number of features) is the high variance of the model, which implies high instability of the prediction.

Consequently, the goal of reducing the dimension is to add more bias to the model, and therefore to penalize the accuracy but to guarantee less variance of the predictions.

The steps of the algorithm consist on creating the principal components from the existing features at first. Then we train the model on the components and transform the PCs back to the original features to make predictions on the actual dataset.

One of the main advantages of performing PCA before regression is to reduce the spatial and temporal complexity of the algorithm, and hence gain in performance. The second is to deal with overfitting especially on data with high colinearity.

The crucial decision to take before performing the PCA is to choose the number of output features. In our model, we choose to select the features depending on a desired level of variance to keep in our data.


## Correlation <a name="correlation"></a>

This technique takes in parameter the number 'k' of features to keep to train the model during next step.
The SelectKbest function using f_regression as score function, compute the correlation between each of the input feature and the label column.
Only the'k' inputs having the higher correlation factors are kept for the model training.

## Step-wise Forward Regression <a name="forward"></a>
forward selection + linear

## Step-wise Backward Regression <a name="backward"></a>
backward selection + linear

## Polynomial Regression <a name="polynomial"></a>
polynomial selection + linear <a name="introduction1"></a>


### Regression Models <a name="models"></a>

## Linear Regression <a name="linear"></a>

Linear Regression fits a linerar model by awarding to each input a coefficient in order to minimize a cost function. In Linear Regression, the cost function is the sum of squared errors between the target and the prediction.

## Ridge Regression <a name="ridge"></a>
Ridge Regression also fits a linear model but with an L2 penalty term being added to the cost function. This penalty term is the sum of the squared of the coefficient multiplied by a constant (Lagrange multiplier)

Ridge regression is a regularized regression, the penalty term being also called a regularization term.

## Lasso Regression <a name="lasso"></a>

Lasso Regression also fits a linear model but with an L1 penalty term being added to the cost function. This penalty term is the sum of the absolute value of the coefficient multiplied by a constant (Lagrange multiplier)

Lasso regression is also a regularized regression.

## Elastic-net Regression <a name="elasticnet"></a>

Elastic-net Regression is a linear regression that combines both L1 and L2 regularization. In fact, the cost function of the Elastic-net model has two penalty terms : one L1 and one L2.

## References <a name="references"></a>

PCA
[] https://en.wikipedia.org/wiki/Principal_component_regression
https://iq.opengenus.org/principal-component-regression/

Prostate Cancer
[] https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data

Boston Housing
[] https://www.kaggle.com/altavish/boston-housing-dataset
