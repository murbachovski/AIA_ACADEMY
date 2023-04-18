import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, load_digits, fetch_covtype

import warnings
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler
warnings.filterwarnings(action='ignore')

datasets = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
]
data_name = [
    'iris',
    'cancer',
    'wine',
]

scaler_list = [
    RobustScaler(),
    MinMaxScaler(),
    MaxAbsScaler(),
    StandardScaler()
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.

#1. DATA
for index, value in enumerate(datasets):
    x, y = value
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        random_state=22,
        test_size=0.25
    )
#2. MODEL
    allAlgorithms = all_estimators(type_filter='classifier')

    max_score = 0
    max_name = '바보'
    for (name, algorithm) in allAlgorithms:
        try:
            model = algorithm()
            
            scores = cross_val_score(model, x_train, y_train, cv=kfold)
            y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
            results = round(np.mean(scores), 4)
            acc = accuracy_score(y_test, y_predict)
            
            if max_score < results:
               max_score = results
               max_name = name
            print('~ing', data_name[index], data_name[value])
        except:
            continue

    print('=================', data_name[index], '=================')
    print('bestModel_Is:', max_name, '\n' 'max_score: ',  max_score)
    print('acc: ', acc)
    print('scores의 평균값 = results: ', results)
    

# y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
# acc = accuracy_score(y_test, y_predict)

# ================= iris =================
# bestModelIs: LinearDiscriminantAnalysis 0.9735
# acc:  0.8421052631578947
# ================= cancer =================
# bestModelIs: GradientBoostingClassifier 0.9718
# acc:  0.9090909090909091
# ================= wine =================
# bestModelIs: RidgeClassifier 0.9926
# acc:  0.6888888888888889