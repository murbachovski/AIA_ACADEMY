# 저장할때 평가결과, 훈련시간을 파일에 넣어줘


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
 
#1. DATA
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) #(20640, 8) (20640,)

#2. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=222, 
)
print(x_train.shape, y_train.shape) #(16512, 8) (16512,)
print(x_test.shape, y_test.shape)   #(4128, 8) (4128,)

#. SCALER 이곳에서 한번만 해줍니다.
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)

# 3. MODEL
model = Sequential()
model.add(Dense(12, input_dim=x.shape[1])) #input_dim=13
model.add(Dense(6))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

# 4. COMPILE
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

import datetime #시간을 저장해주는 놈
date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    # restore_best_weights=True,
    verbose=1
)
mcp = ModelCheckpoint( #restore_best_weights=True가 들어가 있는 모델
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath= "".join([filepath, 'k27_', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=10000, validation_split=0.2, batch_size=50, callbacks=[es, mcp])

# model.save('./_save/MCP/keras27_3save_model.h5')

#5. PREDICT
print("=============================== 1. 기본 출력=======================")
results = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss:', results[0], 'acc:', results[1], 'r2:', r2)

# print("=============================== 2. load_model 출력=======================")
# model2 = load_model('./_save/MCP/keras27_3save_model.h5')
# results = model2.evaluate(x_test, y_test, verbose=0)
# y_predict = model2.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('loss_load:', results[0], 'acc_load:', results[1], 'r2_load:', r2)

# print("=============================== 3. MCP 출력=======================")
# model3 = load_model('./_save/MCP/keras27_3MCP.hdf5')
# results = model3.evaluate(x_test, y_test, verbose=0)
# y_predict = model3.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('loss_load:', results[0], 'acc_load:', results[1], 'r2_load:', r2)