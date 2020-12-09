Dataset datasets/prostate_cancer.csv loaded and cleaned (97 samples)
Preprocessing correlation applied on data
Linear model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.41 |  0.39 |
| CV 2   |  0.62 | -0.19 |
| CV 3   |  0.3  |  0.13 |
| CV 4   |  0.25 |  0.65 |
| CV 5   |  0.47 | -0.03 |
| median |  0.41 |  0.13 |
| mean   |  0.41 |  0.18 |
| std    |  0.12 |  0.28 |

Lasso model results :
|        |   MSE |     r2 |
|:-------|------:|-------:|
| CV 1   |  0.7  |  -4.25 |
| CV 2   |  0.65 |  -8.5  |
| CV 3   |  0.64 |  -1.12 |
| CV 4   |  1.06 |  -4.8  |
| CV 5   |  1.24 | -10.72 |
| median |  0.7  |  -4.8  |
| mean   |  0.83 |  -5.7  |
| std    |  0.23 |   3.1  |

Ridge model results :
|        |   MSE |    r2 |
|:-------|------:|------:|
| CV 1   |  0.42 |  0.13 |
| CV 2   |  0.31 |  0.53 |
| CV 3   |  0.66 | -0.1  |
| CV 4   |  0.47 |  0.31 |
| CV 5   |  0.35 |  0.29 |
| median |  0.42 |  0.29 |
| mean   |  0.44 |  0.24 |
| std    |  0.11 |  0.19 |

Elastic-net model results :
|        |   MSE |     r2 |
|:-------|------:|-------:|
| CV 1   |  0.89 |  -4.38 |
| CV 2   |  0.31 |  -0.88 |
| CV 3   |  1.53 | -16.33 |
| CV 4   |  0.88 |  -2.62 |
| CV 5   |  0.62 |  -2.85 |
| median |  0.88 |  -2.85 |
| mean   |  0.85 |  -4.99 |
| std    |  0.37 |   5.17 |

Backward model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.4  | 0.28 |
| CV 2   |  0.43 | 0.26 |
| CV 3   |  0.5  | 0.14 |
| CV 4   |  0.2  | 0.71 |
| CV 5   |  0.44 | 0.33 |
| median |  0.43 | 0.28 |
| mean   |  0.4  | 0.33 |
| std    |  0.09 | 0.18 |

Forward model results :
|        |   MSE |   r2 |
|:-------|------:|-----:|
| CV 1   |  0.46 | 0.26 |
| CV 2   |  0.58 | 0.18 |
| CV 3   |  0.34 | 0.38 |
| CV 4   |  0.24 | 0.4  |
| CV 5   |  0.29 | 0.59 |
| median |  0.34 | 0.38 |
| mean   |  0.38 | 0.37 |
| std    |  0.11 | 0.13 |

Polynomial model results :
|        |    MSE |    r2 |
|:-------|-------:|------:|
| CV 1   | 145.99 | -0.04 |
| CV 2   |  27.51 | -0.33 |
| CV 3   |   3.78 | -0.49 |
| CV 4   |  17.68 | -0.37 |
| CV 5   |   3.15 | -1.38 |
| median |  17.68 | -0.37 |
| mean   |  35.96 | -0.5  |
| std    |  49.93 |  0.42 |

