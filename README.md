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
1. [Results Analysis](#Results)
   1. [Linear Regression](#linear)
   1. [Lasso Regression](#lasso)
   1. [Ridge Regression](#ridge)
   1. [Elastic-Net Regression](#elasticnet)
   1. [Step-wise Forward and Backward Regression](#forward_backward)
   1. [Polynomial Regression](#polynomial)
1. [References](#references)

## Requirements and Installation <a name="requirements_installation"></a>

- python 3.8
- pip version ≥ 20.0

```
pip install -r requirements.txt
```

## Usage <a name="usage"></a>

In the shell, do not forget to add the path to the repository to your `$PYTHONPATH` variable, by running:
```
export PYTHONPATH="path_to_repo/regression_pipeline/"
```


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

### Feature selection <a name="feature_selection"></a>

## Principal Component Analysis - PCA <a name="PCA"></a>

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
Step-wise regression is a training and fit method consisting on choosing features and model inputs in an automatic procedure. The procedure of feature selection is done by adding on eliminating initial features via a statistic t-test criterion.
Forward regression approache starts with an empty set of features, and for each iteration of the feature selection we add the variable whose inclusion gives the most stastistical significance in terms of the fit. We repeat the procedure until there is no feature that has an acceptable statistical significance on the model.
After the selection procedure, a linear regression is processed on the newly created features.

## Step-wise Backward Regression <a name="backward"></a>
Backward regression is, with forward regression, another form of step-wise regression. While the forward starts with an empty dataset, the backward regression starts with all the features and eliminate at each iteration the inputs with the lest significance based on the t-test. We repeat the procedure until there is no features that can be removed by the statistical test and we proceed to a linear regression on the new features.

## Polynomial Regression <a name="polynomial"></a>
Polynomial regression is a regression analysis in which the relationship between input and output variables is modelled as an nth polynomial degree in the inputs. This approach is quite helpful in the case where a linear relationship assumption on the model does not hold. <a name="introduction1"></a>


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

### Results Analysis <a name="results"></a>

## Polynomial Regression <a name="polynomial"></a>

Linear regression is the most widely used approach when it comes to a linear relationship between the dependant variables (features) and independant variables (labels). However, in the case of non linear structures, a straight line doesn't capture most of data information and thus omits patterns in the data. 
The analysis of boston housing dataset shows that the relationship for example between LSTAT & RM with the label has a second degree polynomial behaviour. Consequently, using polynomial features in our model will allow to have less bias (i.e lower mean squared error) and therefore less underfitting. We should note that the polynomial degree of the features has to be choosen carefully in order to avoid high variance and overfitting in our model. 
For all of these reasons, polynomial regression error score are pretty better than linear regression. However, introducing nth power degree of the feature impacts the stability of the model and results in have a quite better r2 score for linear regression.
For prostate cancer dataset, the data exploration shows that there is less polynomial property in the relationship between lpsa (target) and the inputs. Hence, linear regression performs better than polynomial.

## Step-wise Forward and Backward Regression <a name="forward_backward"></a>

Foward and bakcward regression algorithms can be compared to a principal components analysis in terms of dimensionality reduction. However, the difference between the both is that PCA takes the variables that contains most of the features variance and thus information, whereas forward and backward selects variables with the best statistical information significance and data explanation. 
For boston housing dataset example, forward regression selects 11 features for a p-value of 0.05 and backward regression selects 11 features too for a p-value of 0.01. We can notice and both algorithms drop AGE & INDUS features. This can be interpreted as the customers are not sensitive to weather the zone has old occupied buildings or if there is more or less industrial acres in the town.
For both datasets, stepwise regression has always one of the best scores in terms of bias and variance, i.e MSE and r2. Neverthless, one of the criticism that can be adressed to the stepwise algo is that it can be slower in terms of computation on huge datasets. This is because of the consuming procedure of selection that goes through all the features at each selection iteration.

## Linear regression <a name="Linear"></a>
Linear regression gives good result with the prostate_cancer dataset but not so good result with the boston_hounsing dataset.
We notice that with the prostate_cancer dataset,linear regression gives better result when it is us without feature selection, while for the boston_housing dataset, a feature selection help to improve the score (we conisder MSE as the score here). This means that in the prostate_cancer dataset, the variance is more distribued on the different features while in the boston_housing data set, some features retain the major part of the variance.

## Ridge, Lasso & ElasticNet regression <a name="Ridge, Lasso & Elastic Net"></a>

Ridge, Lasso and ElasticNet regression models works on the same principles.
They all Linear regression model with regularization terms. Ridge regression has an L2 regularization term, Lasso regression has an L1 regularization term and Elasticnet has both L1 and L2 regularization terms linked to each other by a proportionnal factor.
Hence we can analyze the results of this three models together.
The first thing that we notice is that for both dataset, Lasso regression is always the less efficent of these three models. 
What's more, Ridge regression is in both case more efficient than ElasticNet regression. Hence we can conlude that the regularization with an L1 term doesn't help to fit these two datasets.


## References <a name="references"></a>

PCA
[] https://en.wikipedia.org/wiki/Principal_component_regression
https://iq.opengenus.org/principal-component-regression/

Polynomial regression 
[] https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

Step-wise regression
[] https://en.wikipedia.org/wiki/Stepwise_regression

Prostate Cancer
[] https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data

Boston Housing
[] https://www.kaggle.com/altavish/boston-housing-dataset
