import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, load_digits, fetch_covtype

import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVC

warnings.filterwarnings(action='ignore')

datasets = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
    load_digits(return_X_y=True),
]
data_name = [
    'iris',
    'cancer',
    'wine',
    'digits',
]

scaler_list = [
    RobustScaler(),
    MinMaxScaler(),
    MaxAbsScaler()
]
model_list = [
    RandomForestRegressor(),
    AdaBoostRegressor()
]

result_list = []

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22)

#1. DATA
for index, value in enumerate(datasets):
    x, y = value

#2. MODEL
    max_score = 0
    max_name = ''
    for scaler in scaler_list:
        for model in model_list:
            try:
                pipeline = make_pipeline(scaler, model)

                scores = cross_val_score(pipeline, x, y, cv=kfold)
                results = round(np.mean(scores), 4)
                scaler_name = scaler.__class__.__name__
                model_name = model.__class__.__name__

                if max_score < results:
                   max_score = results
                   max_name = f'{scaler_name} + {model_name}'
                print('~ing', data_name[index])
            except:
                continue
            
        print('=================', data_name[index], '=================')
        print('bestModelIs:', max_name, max_score)
        result_list.append((data_name[index], max_name, max_score))

# Print all results at the end
print('===============================================')
print('FINAL RESULTS:')
for result in result_list:
    print(f'{result[0]}: {result[1]} ({result[2]})')
