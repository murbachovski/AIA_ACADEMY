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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
    {'n_estimators': [100, 200], 'max_depth' : [6, 10, 12]}
]



#1. DATA
path = './_data/ddarung/' # path ./은 현재 위치
path_save = './_save/ddarung/'
# Column = Header


# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0) 

# TEST
test_csv = pd.read_csv(path + "test.csv",
                        index_col=0) 

train_csv = train_csv.dropna()


x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.1, #stratify=y # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)    

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. MODEL # 대문자 = class
model = RandomizedSearchCV(RandomForestRegressor(),
                            parameters,
                              verbose=1,
                                refit=True,
                                  n_jobs=-1,
                                    cv=5,
                                      n_iter=5 # 디폴트는 = 10 * cv만큼 훈련
                                      )

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

# 최적의 매개변수 :  RandomForestRegressor(max_depth=12)
# 최적의 파라미터 :  {'max_depth': 12, 'n_estimators': 100}
# 최적의 인덱스 :  4
# BEST SCORE :  0.7725553298701187
# model 스코어 :  0.7756651257736299
# 걸린 시간:  6.88 초


# SCALER +  적용 후
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12)
# 최적의 파라미터 :  {'max_depth': 12, 'n_estimators': 100}
# 최적의 인덱스 :  4
# BEST SCORE :  0.7705315274698468
# model 스코어 :  0.7855882974161251
# 걸린 시간:  7.92 초

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 12}
# 최적의 인덱스 :  5
# BEST SCORE :  0.7811080416805739
# model 스코어 :  0.7756078842976654
# 걸린 시간:  7.77 초