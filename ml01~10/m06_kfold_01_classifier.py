import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, load_digits, fetch_covtype

import warnings
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
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
    '디지트',
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.

#1. DATA
for index, value in enumerate(datasets):
    x, y = value

#2. MODEL
    allAlgorithms = all_estimators(type_filter='classifier')

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

# ================= iris =================
# bestModelIs: LinearDiscriminantAnalysis 0.98
# ================= cancer =================
# bestModelIs: ExtraTreesClassifier 0.9701
# ================= wine =================
# bestModelIs: LinearDiscriminantAnalysis 0.9943
# ================= 디지트 =================
# bestModelIs: SVC 0.9872

# SCALER
# ================= iris =================
# bestModelIs: LinearDiscriminantAnalysis 0.98
# ================= cancer =================
# bestModelIs: LogisticRegression 0.9807
# ================= wine =================
# bestModelIs: LinearDiscriminantAnalysis 0.9943
# ================= 디지트 =================
# bestModelIs: ExtraTreesClassifier 0.9833