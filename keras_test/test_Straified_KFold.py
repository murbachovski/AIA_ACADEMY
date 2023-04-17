import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict,cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#1. DATA
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=123,
    shuffle=True,
    test_size=0.2,
)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

#2. MODEL
model = SVC()

#3. 4. COMPILE PREDICT
scores = cross_val_score(model, x_train, y_train, cv=kfold)
y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_predict)
print('CROSS_VAL_SCORE: ', scores, '\n 교차 검증 평균 점수: ', round(np.mean(scores), 4), '\n CROSS_VAL_PREDICT_ACC: ', acc)
# CROSS_VAL_SCORE:  [1.         0.95833333 0.95833333 1.         0.95833333] 
#  교차 검증 평균 점수:  0.975
#  CROSS_VAL_PREDICT_ACC:  0.8333333333333334
