import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, load_digits, fetch_covtype

import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV

warnings.filterwarnings(action='ignore')

# Load datasets
datasets = [
    load_iris(return_X_y=True)
]
data_name = [
    'iris'
]

# Scalers to use
scaler_list = [
    RobustScaler(),
    MinMaxScaler(),
    MaxAbsScaler()
]

# Models to use
model_list = [
    RandomForestClassifier(),
]

# Hyperparameters for each model
parameters = [
    {'randomforestclassifier__n_estimators': [100, 200], 'randomforestclassifier__max_depth' : [6, 10, 12]}
]

# Grids to use for hyperparameter tuning
grid_list = [GridSearchCV(model_list, parameters), RandomizedSearchCV(model_list, parameters), HalvingGridSearchCV(model_list, parameters)]
grid_names = ['GridSearchCV', 'RandomizedSearchCV', 'HalvingGridSearchCV', 'GridSearchCV with pipeline']

# List to store results
result_list = []

# Cross-validation settings
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22)

# Loop through datasets
for index, value in enumerate(datasets):
    x, y = value

    # Find best model/scaler combination for current dataset
    max_score = 0
    max_name = ''
    for scaler in scaler_list:
        for model in model_list:
            for i, grid in enumerate(grid_list):
                try:
                    grid_cv = grid
                    pipeline = make_pipeline(scaler, model)

                    scores = cross_val_score(pipeline, x, y, cv=kfold)
                    results = round(np.mean(scores), 4)
                    scaler_name = scaler.__class__.__name__
                    model_name = model.__class__.__name__

                    if max_score < results:
                        max_score = results
                        max_name = f'{scaler_name} + {model_name} ({grid_names[i]})'
                    print('~ing', data_name[index])
                except:
                    continue
            
            # Print best model/scaler combination for current dataset
            print('=================', data_name[index], '=================')
            print('bestModelIs:', max_name, max_score)
            result_list.append((data_name[index], max_name, max_score))

# Print all results at the end
print('===============================================')
print('FINAL RESULTS:')
for result in result_list:
    print(f'{result[0]}) {result[1]} ({result[2]})')
