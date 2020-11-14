# Linear regression pipeline


## Requirements :

- python 3.8
- pip version ≥ 20.0

## Installation :

```
pip install -r requirements.txt
```

## Usage

```
python main.py dataset_filename
```

## Datasets

### Boston Housing:

Download this dataset [here](https://www.kaggle.com/altavish/boston-housing-dataset).

There are 13 features to predict the price houses in Boston.

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
  - `B`: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
  - `LSTAT`: % lower status of population.
  - `MEDV` (target): median value of owner-occupied homes in $1000's.


### Prostate Cancer:

Download this dataset [here](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/prostate.data).
The last column of the original dataset `train` has been removed.

There are 8 features to predict a log of PSA (Prostate Specific Antigen) value, a marker for Prostate cancer.

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
