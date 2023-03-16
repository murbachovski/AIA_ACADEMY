from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
                                                

#1. DATA
dataset = load_diabetes()
x = dataset.data
y = dataset.target
# print(type(x))
# print(x)

# #정규화 변환
# print(np.min(x), np.max(x))
# MinMaxScaler()                # loss:  27289.775390625
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(np.min(x), np.max(x))    
# StandardScaler()              # loss:  27461.326171875
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# MaxAbsScaler()                # loss:  27758.453125
# scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# scaler = RobustScaler()         # loss:  27515.703125
# scaler.fit(x)
# x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, 
    random_state=333
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

#2. MODEL
model = Sequential()
model.add(Dense(1, input_dim=x.shape[1]))

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)