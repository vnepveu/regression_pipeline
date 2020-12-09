Dataset datasets/boston_housing.csv loaded and cleaned (506 samples)
Preprocessing correlation applied on data
Linear model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.37 | 0.4  |
| CV 2   |  0.28 | 0.56 |
| CV 3   |  0.29 | 0.68 |
| CV 4   |  0.29 | 0.54 |
| CV 5   |  0.48 | 0.2  |
| median |  0.29 | 0.54 |
| mean   |  0.33 | 0.48 |
| std    |  0.07 | 0.15 |

Lasso model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.57 | -1.8  |
| CV 2   |  0.85 | -2.12 |
| CV 3   |  0.9  | -4.83 |
| CV 4   |  0.78 | -2.47 |
| CV 5   |  0.84 | -3.13 |
| median |  0.84 | -2.47 |
| mean   |  0.8  | -2.8  |
| std    |  0.11 |  0.99 |

Ridge model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.27 | 0.59 |
| CV 2   |  0.28 | 0.59 |
| CV 3   |  0.53 | 0.39 |
| CV 4   |  0.37 | 0.43 |
| CV 5   |  0.29 | 0.41 |
| median |  0.29 | 0.43 |
| mean   |  0.34 | 0.47 |
| std    |  0.09 | 0.08 |

Elastic-net model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.52 | -1.31 |
| CV 2   |  0.55 | -1.47 |
| CV 3   |  1.14 | -3.93 |
| CV 4   |  1.04 | -4    |
| CV 5   |  0.51 | -1.62 |
| median |  0.55 | -1.62 |
| mean   |  0.72 | -2.32 |
| std    |  0.26 |  1.17 |

Backward model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.25 | 0.66 |
| CV 2   |  0.37 | 0.44 |
| CV 3   |  0.37 | 0.37 |
| CV 4   |  0.42 | 0.38 |
| CV 5   |  0.29 | 0.58 |
| median |  0.37 | 0.44 |
| mean   |  0.35 | 0.48 |
| std    |  0.06 | 0.11 |

Forward model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.42 | 0.47 |
| CV 2   |  0.28 | 0.58 |
| CV 3   |  0.2  | 0.67 |
| CV 4   |  0.54 | 0.09 |
| CV 5   |  0.26 | 0.61 |
| median |  0.28 | 0.58 |
| mean   |  0.33 | 0.5  |
| std    |  0.11 | 0.19 |

Polynomial model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  1.16 | 0.47 |
| CV 2   |  0.14 | 0.82 |
| CV 3   |  0.33 | 0.65 |
| CV 4   |  0.19 | 0.79 |
| CV 5   |  0.78 | 0.39 |
| median |  0.33 | 0.65 |
| mean   |  0.49 | 0.63 |
| std    |  0.36 | 0.16 |

