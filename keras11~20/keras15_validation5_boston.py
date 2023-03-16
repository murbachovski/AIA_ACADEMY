#0. IMPORT
#1. DATA(train_test_split()로 validation 뽑아내기.)
#2. MODEL
#2. COMPILE
#3. EVALUATE, PREDICT

import numpy as np      
import pandas as pd
import tensorflow as tf 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error # MSE

#1. DATA
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

#1-2. train_test_split()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.6,
    random_state=32
)
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.3,
    random_state=32
)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim = 13))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 32, validation_data = (x_val, y_val))

#4. VALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

#5. DEF
def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)