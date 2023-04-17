import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#1. DATA
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2, #stratify=y
)

n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

#2. MODEL
model = SVC()

#3. 4. COMPILE, PREDICT
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)
print('CROSS_VAL_SCORE : ', score, '\n 교차 검증 평균 점수 : ', round(np.mean(score),4 ), '\n CROSS_VAL_PREDICT_ACC: ', acc)
print('====================================')
# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_test, return_counts=True))

# SVC
# CROSS_VAL_SCORE :  [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333] 
#  교차 검증 평균 점수 :  0.9583
#  ACC:  0.8333333333333334
# SVC / stratify=y
# CROSS_VAL_SCORE :  [1.         1.         0.91666667 0.83333333 0.91666667] 
#  교차 검증 평균 점수 :  0.9333
#  CROSS_VAL_PREDICT_ACC:  0.9

# RandomForestClassifier
# CROSS_VAL_SCORE :  [0.95833333 0.95833333 1.         0.95833333 0.91666667] 
#  교차 검증 평균 점수 :  0.9583
#  CROSS_VAL_PREDICT_ACC:  1.0

