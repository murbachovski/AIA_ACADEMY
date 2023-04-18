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
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import time

# parameters = [
#     {'n_estimators' : [100, 200]},
#     {'max_depth' : [6, 8, 10, 12]},
#     {'min_samples_leaf': [3, 5, 7, 10]},
#     {'min_samples_split' : [2, 3, 5, 10]},
#     {'n_jobs' : [-1, 2, 4]},
# ]

parameters = [
    {'n_estimators': [100, 200], 'max_depth' : [6, 10, 12], 'min_samples_split' : [2, 3, 5, 10]}
]



#1. DATE
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv =  pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL(결측치 처리)
train_csv = train_csv.dropna()

#1-5. (x, y DATA SPLIT)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=32
)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(7620, 8, 1)
x_test = x_test.reshape(3266, 8, 1)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, #stratify=y # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)


#2. MODEL # 대문자 = class
model = GridSearchCV(RandomForestRegressor(), parameters, verbose=1, refit=True, n_jobs=-1, cv=5)

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

# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_estimators=200)
# 최적의 파라미터 :  {'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 200}
# 최적의 인덱스 :  9
# BEST SCORE :  0.355212266607574
# model 스코어 :  0.36736257502102887
# 걸린 시간:  45.79 초
