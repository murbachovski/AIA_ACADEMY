import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_wine, fetch_covtype
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt

data_lsit = [
    load_iris(return_X_y=True),
    load_digits(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
    load_digits(return_X_y=True),
    fetch_covtype(return_X_y=True)
]

data_name_list = [
    'load_iris',
    'load_digits',
    'load_breast_cancer',
    'load_wine',
    'load_digits',
    'fetch_covtype'
]

# # 1. DATA
for i, v in enumerate(data_lsit):
    x, y = v
for n in data_name_list:
    lda = LinearDiscriminantAnalysis()
    x = lda.fit_transform(x, y)
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print((f'DATA_NAME: {n}'), cumsum)
   
# 데이터:  load_iris [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# 데이터:  load_digits [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# 데이터:  load_breast_cancer [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# 데이터:  load_wine [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# 데이터:  load_digits [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]
# 데이터:  fetch_covtype [0.72159835 0.89235761 0.94351071 0.9742032  0.99067616 1.        ]