from sklearn.datasets import load_diabetes

#1. DATA
dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape, y.shape) # (442, 10) (442,) input_dim = 10 

#[실습]
# R2 = 0.62 이상

#2. MODEL
import numpy as np
import tensorflow as tf     
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

model = Sequential()
model.add(Dense(256, input_dim = 10))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

input1 = Input(shape=(10,))
dense1 = Dense(256)(input1)
dense2 = Dense(128)(dense1)
dense3 = Dense(64)(dense2)
dense4 = Dense(32)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)


# input1 = Input(shape = (8,))
# dense1 = Dense(30)(input1)
# dense2 = Dense(20)(dense1)
# dense3 = Dense(10)(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)


#3. COMPILE
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.6,
    random_state=123
)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 100, validation_split=0.3)
# from tensorflow.python.keras.models import Sequential
# 이렇게 import해줘야 validation_split 자동완성 나옵니다.

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) # GOOD, Y = wX + b, 보조지표
print("R2_SCORE: ", r2)

#r2_SCORE = 0.51
#R2_SCORE:  0.6432587748923371