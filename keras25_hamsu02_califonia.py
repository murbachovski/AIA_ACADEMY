from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
                                                

#1. DATA
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target
# print(type(x))
# print(x)

# #정규화 변환
scaler = RobustScaler()
scaler.fit(x)
x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, 
    random_state=333
)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test)) 

#2. MODEL
# model = Sequential()
# model.add(Dense(30, input_shape=(8,)))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))

input1 = Input(shape = (8,))
dense1 = Dense(30)(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(1)(dense3)
model = Model(inputs=input1, outputs=output1)

# 데이터가 3차원이면(시계열 데이터)
# (1000, 100, 1) ==> input_shape=(100, 1)

# 데이터가 4차원이면(이미지 데이터)
# (60000, 32, 32, 3) ==> input_shape=(32, 32, 3)
 


#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)