#<과적합 배제>
# 데이터를 많이 넣는다.
# 노드의 일부를 뺀다. dropout

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import matplotlib.pyplot as plt
 
#1. DATA
dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) #(569, 30) (569,)

#2. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=222, 
)
print(x_train.shape, y_train.shape) #(455, 30) (455,)
print(x_test.shape, y_test.shape)   #(114, 30) (114,)

#. SCALER 이곳에서 한번만 해줍니다.
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) #(455, 30) (114, 30)

# 3. MODEL
# input1 = Input(shape = (13,))
# dense1 = Dense(30)(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(20)(dense1)
# drop2 = Dropout(0.2)(dense2)
# dense3 = Dense(10)(drop2)
# drop3 = Dropout(0.2)(dense3)
# output1 = Dense(1)(drop3)
# model = Model(inputs=input1, outputs=output1)

model = Sequential()
model.add(Dense(30, input_shape=(30, ))) #input_dim=13
model.add(Dropout(0.3))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
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
    filepath= "".join([filepath, 'k28_dropout_breast_cancer', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=10000, validation_split=0.2, batch_size=50, callbacks=[es, mcp]) #mcp

# model.save('./_save/MCP/keras27_3save_model.h5')

model2 = load_model('_save\MCP\keras27_4\k28_dropout_breast_cancer0314_1253_0149-0.0709.hdf5')

#5. PREDICT
print("=============================== 1. 기본 출력=======================")
results = model.evaluate(x_test, y_test, verbose=0)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss:', results[0], 'acc:', results[1], 'r2:', r2)
# loss: 0.06383825093507767 acc: 0.9824561476707458 r2: 0.730635716124379

print("=============================== 1. MCP 출력=======================")
results = model2.evaluate(x_test, y_test, verbose=0)
y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss:', results[0], 'acc:', results[1], 'r2:', r2)
# loss: 0.06425245106220245 acc: 0.9736841917037964 r2: 0.7288880323216033