# 23.03.06 배운 것들 천천히 생각해보며 정리
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # 모델 설정
from tensorflow.keras.layers import Dense # 히든 레이아웃
from sklearn.model_selection import train_test_split # x, y 나눌때
from sklearn.metrics import r2_score, mean_absolute_error # MSE
from sklearn.datasets import load_boston # 데이터 불러올때

#1. DATA
path = "./_data/ddarung/"

#1-1. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
print(train_csv)
print(train_csv.shape)
print(train_csv.columns)
print(train_csv.describe())

# #1-2. TEST
# test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)
# print(test_csv.shape)
# print(test_csv.columns)
# print(test_csv.describe())

# #1-3. DROP(결측치 처리)
# train_csv = train_csv.dropna()

# #1-4. x, y 데이터 분리
# x = train_csv.drop(['count'], axis=1)
# y = train_csv['count']

# #1-5. Shuffle
# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     train_size=0.7,
#     random_state=10
# )

# #2. MODEL
# model = Sequential()
# model.add(Dense(256, input_dim = 9))
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(1))

# #3. COMPILE
# model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(x_train, y_train, epochs = 100, batch_size = 32, verbose = 1)

# #4. EVALUATE, PREDICT
# loss = model.evaluate(x_test, y_test)
# print('loss:', loss)
# y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2:', r2)

# #5. DEF RMSE
# def RMSE(y_test, y_predcit):
#     return np.sqrt(mean_absolute_error(y_test, y_predcit))
# rmse = RMSE(y_test, y_predict)
# print('rmse:', rmse)

# #6. submission
# y_submit = model.predict(test_csv)
# submission = pd.read_csv(path + 'submission.csv', index_col=0)
# submission['count'] = y_submit
# submission.to_csv(path + 'submit_2303062116.csv')