Dataset datasets/prostate_cancer.csv loaded and cleaned (97 samples)
Preprocessing pca applied on data
Linear model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.39 |  0.19 |
| CV 2   |  0.36 |  0.38 |
| CV 3   |  0.57 |  0.06 |
| CV 4   |  0.56 |  0.38 |
| CV 5   |  0.47 | -0.22 |
| median |  0.47 |  0.19 |
| mean   |  0.47 |  0.16 |
| std    |  0.08 |  0.2  |

Lasso model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.8  | -5.65 |
| CV 2   |  0.46 | -7.86 |
| CV 3   |  0.78 | -2.07 |
| CV 4   |  1.46 | -6.47 |
| CV 5   |  0.68 | -7.24 |
| median |  0.78 | -6.47 |
| mean   |  0.82 | -5.96 |
| std    |  0.31 |  1.87 |

Ridge model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.54 |  0.12 |
| CV 2   |  0.36 |  0.2  |
| CV 3   |  0.4  | -0.02 |
| CV 4   |  0.74 | -0.2  |
| CV 5   |  0.35 |  0.34 |
| median |  0.4  |  0.12 |
| mean   |  0.46 |  0.09 |
| std    |  0.14 |  0.17 |

Elastic-net model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.34 | -3.32 |
| CV 2   |  0.83 | -4.42 |
| CV 3   |  0.84 | -5.01 |
| CV 4   |  0.62 | -1.42 |
| CV 5   |  0.81 | -2.13 |
| median |  0.81 | -3.32 |
| mean   |  0.71 | -3.27 |
| std    |  0.18 |  1.23 |

Backward model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.57 | -0.23 |
| CV 2   |  0.36 |  0.31 |
| CV 3   |  0.27 |  0.51 |
| CV 4   |  0.65 | -0.26 |
| CV 5   |  0.44 |  0.23 |
| median |  0.44 |  0.23 |
| mean   |  0.45 |  0.13 |
| std    |  0.13 |  0.28 |

Forward model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.44 | 0.29 |
| CV 2   |  0.54 | 0.17 |
| CV 3   |  0.48 | 0.22 |
| CV 4   |  0.41 | 0.2  |
| CV 5   |  0.31 | 0.35 |
| median |  0.44 | 0.22 |
| mean   |  0.44 | 0.24 |
| std    |  0.07 | 0.06 |

Polynomial model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   | 48.47 |  0.06 |
| CV 2   | 22.46 |  0.02 |
| CV 3   |  3.42 |  0.15 |
| CV 4   |  7.2  |  0.05 |
| CV 5   |  4.85 | -0.37 |
| median |  7.2  |  0.05 |
| mean   | 15.6  | -0    |
| std    | 15.98 |  0.17 |

