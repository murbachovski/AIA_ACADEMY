import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, load_digits

import warnings
warnings.filterwarnings(action='ignore')

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import RobustScaler

datasets = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
]

data_name = [
    'load_iris',
    'load_breast_cancer',
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)

#1. DATA
for i, v in enumerate(datasets):
    x, y = v
    #2. MODEL
    allAlgoritms = all_estimators(type_filter='classifier')

    max_score = 0
    max_name = ''
    for (name, algorithm) in allAlgoritms:
        try:
            model = algorithm()

            scores = cross_val_score(model, x, y, cv=kfold)
            results = round(np.mean(scores), 4)

            if max_score < results:
                max_score = results
                max_name = name
                print('~ing')
        except:
            continue

    print('======================', data_name[i], '=================')
    print('BestModelIs : ', max_name, max_score)