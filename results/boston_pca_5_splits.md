Dataset datasets/boston_housing.csv loaded and cleaned (506 samples)
Preprocessing pca applied on data
Linear model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.26 | 0.67 |
| CV 2   |  0.27 | 0.6  |
| CV 3   |  0.46 | 0.33 |
| CV 4   |  0.24 | 0.62 |
| CV 5   |  0.33 | 0.51 |
| median |  0.27 | 0.6  |
| mean   |  0.31 | 0.55 |
| std    |  0.07 | 0.11 |

Lasso model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.63 | -2.46 |
| CV 2   |  0.62 | -1.88 |
| CV 3   |  0.75 | -2.4  |
| CV 4   |  0.83 | -4.37 |
| CV 5   |  1.11 | -2.94 |
| median |  0.75 | -2.46 |
| mean   |  0.78 | -2.75 |
| std    |  0.16 |  0.79 |

Ridge model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.35 | 0.39 |
| CV 2   |  0.32 | 0.6  |
| CV 3   |  0.25 | 0.66 |
| CV 4   |  0.47 | 0.26 |
| CV 5   |  0.2  | 0.71 |
| median |  0.32 | 0.6  |
| mean   |  0.32 | 0.54 |
| std    |  0.09 | 0.16 |

Elastic-net model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.75 | -2.04 |
| CV 2   |  0.79 | -2.74 |
| CV 3   |  0.67 | -1.99 |
| CV 4   |  0.39 | -0.7  |
| CV 5   |  0.86 | -3.5  |
| median |  0.75 | -2.04 |
| mean   |  0.7  | -2.17 |
| std    |  0.15 |  0.85 |

Backward model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.26 | 0.67 |
| CV 2   |  0.25 | 0.58 |
| CV 3   |  0.4  | 0.34 |
| CV 4   |  0.34 | 0.43 |
| CV 5   |  0.31 | 0.66 |
| median |  0.31 | 0.58 |
| mean   |  0.31 | 0.54 |
| std    |  0.05 | 0.12 |

Forward model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.25 | 0.71 |
| CV 2   |  0.23 | 0.65 |
| CV 3   |  0.38 | 0.38 |
| CV 4   |  0.37 | 0.44 |
| CV 5   |  0.33 | 0.52 |
| median |  0.33 | 0.52 |
| mean   |  0.31 | 0.54 |
| std    |  0.06 | 0.11 |

Polynomial model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.25 | 0.75 |
| CV 2   |  0.77 | 0.43 |
| CV 3   |  0.79 | 0.22 |
| CV 4   |  0.15 | 0.83 |
| CV 5   |  0.5  | 0.47 |
| median |  0.5  | 0.47 |
| mean   |  0.49 | 0.53 |
| std    |  0.24 | 0.21 |

