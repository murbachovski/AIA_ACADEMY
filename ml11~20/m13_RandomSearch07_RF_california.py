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
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
model = RandomizedSearchCV(RandomForestRegressor(), parameters, verbose=1, refit=True, n_jobs=-1, cv=5)

#3. COMPILE
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print("최적의 매개변수 : ", model.best_estimator_) 

print("최적의 파라미터 : ", model.best_params_)

print("최적의 인덱스 : ", model.best_index_)

print("BEST SCORE : ", model.best_score_)
print("model 스코어 : ", model.score(x_test, y_test))

print('걸린 시간: ', round(end_time - start_time, 2),'초')

# GridSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=6)
# 최적의 파라미터 :  {'max_depth': 6, 'n_estimators': 100}
# 최적의 인덱스 :  0
# BEST SCORE :  0.9377363836643788
# model 스코어 :  0.8717649006622517
# 걸린 시간:  4.17 초

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 12}
# 최적의 인덱스 :  5
# BEST SCORE :  0.9355914965137047
# model 스코어 :  0.8869719205298013
# 걸린 시간:  3.93 초