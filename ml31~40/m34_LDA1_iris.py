import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine, fetch_covtype
from sklearn.metrics import accuracy_score

# 1. DATA
data_lsit = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
    load_digits(return_X_y=True),
    # fetch_covtype(return_X_y=True)
]

data_name_list = [
    'load_iris',
    'load_breast_cancer',
    'load_wine',
    'load_digits',
    # 'fetch_covtype'
]

print("===========시작==============")
for i, (x, y) in enumerate(data_lsit):
    model = RandomForestClassifier()
    model.fit(x, y)
    results = model.score(x, y)
    print("=======BEFORE========")
    print('results:', results)
    
    print("=======AFTER_LinearDiscriminantAnalysis========")
    lda = LinearDiscriminantAnalysis()
    x_lda = lda.fit_transform(x, y)

    model.fit(x_lda, y)
    results = model.score(x_lda, y)
    print('results:', results)
    
    print(data_name_list[i], x.shape, '=>', x_lda.shape)
    lda_EVR = lda.explained_variance_ratio_
    cumsum = np.cumsum(lda_EVR)
    print(cumsum)
print("===========끝==============")


# =======BEFORE========
# results: 1.0
# =======AFTER_LinearDiscriminantAnalysis========
# results: 1.0
# load_iris (150, 4) => (150, 2)
# [0.9912126 1.       ]
# =======BEFORE========
# results: 1.0
# =======AFTER_LinearDiscriminantAnalysis========
# results: 1.0
# load_digits (569, 30) => (569, 1)
# [1.]
# =======BEFORE========
# results: 1.0
# =======AFTER_LinearDiscriminantAnalysis========
# results: 1.0
# load_breast_cancer (178, 13) => (178, 2)
# [0.68747889 1.        ]
# =======BEFORE========
# results: 1.0
# =======AFTER_LinearDiscriminantAnalysis========
# results: 1.0
# load_wine (1797, 64) => (1797, 9)
# [0.28912041 0.47174829 0.64137175 0.75807724 0.84108978 0.90674662
#  0.94984789 0.9791736  1.        ]

