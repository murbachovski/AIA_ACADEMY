# 회귀 데이터 싹 모아서 테스트

import numpy as np
from sklearn.datasets import fetch_california_housing

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error # MSE
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
import datetime #시간을 저장해주는 놈
from sklearn.preprocessing import MinMaxScaler

date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/kaggle_bike/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL(결측치 처리)
train_csv = train_csv.dropna()

#1-5. (x, y DATA SPLIT)
x1 = train_csv.drop(['count', 'casual', 'registered'], axis=1)   #?
y1 = train_csv['count']

#1-6. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x1,
    y1,
    test_size=0.3,
    random_state=32
)

#1. DATA
x, y = fetch_california_housing(return_X_y=True)

#2. MODEL
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

from sklearn.svm import LinearSVC # 분류

from sklearn.linear_model import LogisticRegression # 분류

from sklearn.tree import DecisionTreeRegressor # 회귀

from sklearn.tree import DecisionTreeClassifier # 분류

from sklearn.ensemble import RandomForestRegressor # 회귀

from sklearn.ensemble import RandomForestClassifier # 분류

# model = LinearSVC(C=500) # 알고리즘 연산이 포함되어 있다. C가 클수록 곡선을 그려준다. 작을수록 직선

# model = LogisticRegression() # 이진 분류 == sigmoid

# model = DecisionTreeRegressor() #회귀 모델
# model1 = DecisionTreeRegressor()

# model = DecisionTreeClassifier()

model = RandomForestRegressor()
model1 = RandomForestRegressor()

# model = RandomForestClassifier()

#3. COMPILE
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) # 원핫 안 해줬을 경우 가능하다. 0부터 시작하는지 확인하기
# model.fit(x,y, epochs=100, validation_split=0.2)
model.fit(x,y)
model1.fit(x_train, y_train)

#4. PREDICT
# results = model.evaluate(x, y)
results = model.score(x,y)
results1 = model1.score(x_train, y_train)
print('califonia:', results)
print('kaggle_bike:', results1)

# califonia: 0.9742103426288854
# kaggle_bike: 0.8562310493188482