import numpy as np
# from sklearn.datasets import load_iris, load_breast_cancer
# from sklearn.datasets import load_wine, load_digits, fetch_covtype
from sklearn.datasets import load_diabetes, fetch_california_housing

import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
warnings.filterwarnings(action='ignore')

datasets = [
    load_diabetes(return_X_y=True),
    fetch_california_housing(return_X_y=True)
]

data_name = [
    'load_diabetes',
    'fetch_california_housing'
]

scaler_list = [
    RobustScaler(),
    MinMaxScaler(),
    StandardScaler(),
    MaxAbsScaler()
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22)

#1. DATA
for i, v in enumerate(datasets):
    x, y = v
    for s, b in enumerate(scaler_list):
        x = b

    #2. MODEL
        allAlgorithms = all_estimators(type_filter='regressor')
        max_score = 0
        max_name = 'test'
        for (name, algorithm) in allAlgorithms:
            try:
                model = algorithm()
                scores = cross_val_score(model, x, y, cv=5)
                results = round(np.mean(scores), 4)

                if max_score < results:
                    max_score = results
                    max_name = name
                    print('-ing')
            except:
                continue

    print('=============', data_name[i],'=============')
    print('bestModel_Is:', max_name, max_score)