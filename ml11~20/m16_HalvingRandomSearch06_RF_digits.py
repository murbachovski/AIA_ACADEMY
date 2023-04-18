# 01. iris
# 02. cancer
# 03. dacon_diabets
# 04. wine

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
import time

# parameters = [
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf': [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]},
# ]

parameters = [
    {'n_estimators': [100, 200], 'max_depth' : [6, 10, 12]}
]

# 1. 데이터
x,y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, #stratify=y # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


#2. MODEL # 대문자 = class
model = HalvingGridSearchCV(RandomForestClassifier(), parameters, verbose=1, refit=True, n_jobs=-1, cv=5)

#3. COMPILE
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_) 

print("최적의 파라미터 : ", model.best_params_)

print("최적의 인덱스 : ", model.best_index_)

print("BEST SCORE : ", model.best_score_)
print("model 스코어 : ", model.score(x_test, y_test))


y_predict = model.predict(x_test)
print('ACC: ', accuracy_score(y_test, y_predict))
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# 최적의 인덱스 :  0
# BEST SCORE :  0.9916666666666668
# model 스코어 :  1.0
# ACC:  1.0

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠ACC: ', accuracy_score(y_test, y_pred_best))
# 최적 튠ACC:  1.0

print('걸린 시간: ', round(end_time - start_time, 2),'초')

# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 200}
# 최적의 인덱스 :  1
# BEST SCORE :  0.9862068965517242
# model 스코어 :  0.9722222222222222
# ACC:  0.9722222222222222
# 최적 튠ACC:  0.9722222222222222
# 걸린 시간:  4.13 초

# HalvingGridSearchCV
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12)
# 최적의 파라미터 :  {'max_depth': 12, 'n_estimators': 100}
# 최적의 인덱스 :  6
# BEST SCORE :  0.9928571428571429
# model 스코어 :  0.9444444444444444
# ACC:  0.9444444444444444
# 최적 튠ACC:  0.9444444444444444
# 걸린 시간:  4.71 초