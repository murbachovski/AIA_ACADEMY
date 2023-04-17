import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#1. DATA
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=337, test_size=0.2
)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

#2. MODEL
model = RandomForestClassifier()

#3. 4. COMPILE, PREDICT
score = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)
print('CROSS_VAL_SCORE : ', score, '\n 교차 검증 평균 점수 : ', round(np.mean(score),4 ), '\n CROSS_VAL_PREDICT_ACC: ', acc)

# SVC
# CROSS_VAL_SCORE :  [0.95833333 0.95833333 0.95833333 0.95833333 0.95833333] 
#  교차 검증 평균 점수 :  0.9583
#  ACC:  0.8333333333333334

# RandomForestClassifier
# CROSS_VAL_SCORE :  [0.95833333 0.95833333 1.         0.95833333 0.91666667] 
#  교차 검증 평균 점수 :  0.9583
#  CROSS_VAL_PREDICT_ACC:  1.0