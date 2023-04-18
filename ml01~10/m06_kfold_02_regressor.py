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
    fetch_california_housing(return_X_y=True),
]
data_name = [
    'load_diabetes',
    'fetch_california_housing'
]
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.

#1. DATA
for index, value in enumerate(datasets):
    x, y = value
    scaler = RobustScaler()
    x = scaler.fit_transform(x)

#2. MODEL
    allAlgorithms = all_estimators(type_filter='regressor')

    max_score = 0
    max_name = '바보'
    for (name, algorithm) in allAlgorithms:
        try:
            model = algorithm()
            
            scores = cross_val_score(model, x, y, cv=kfold)
            results = round(np.mean(scores), 4)
            
            if max_score < results:
               max_score = results
               max_name = name
            print('~ing', data_name[index], data_name[value])
        except:
            continue
        
    print('=================', data_name[index], '=================')
    print('bestModelIs:', max_name, max_score)