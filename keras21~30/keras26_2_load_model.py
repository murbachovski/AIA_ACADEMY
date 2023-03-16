from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
                                                

#1. DATA
dataset = load_boston()
x = dataset.data
y = dataset.target

# SCALER
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)

# TRAIN_TEST
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, 
    random_state=333
)

# SCALER
scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
#위에 두 줄과 같은 한 줄
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. MODEL
# input1 = Input(shape = (13,))
# dense1 = Dense(30)(input1)
# dense2 = Dense(20)(dense1)
# dense3 = Dense(10)(dense2)
# output1 = Dense(1)(dense3)
# model = Model(inputs=input1, outputs=output1)
# model.save('./_save/keras26_1_save_model.h5')

model = load_model('./_save/keras26_1_save_model.h5') #모델의 구조만 저장되어 있다.
model.summary()

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)