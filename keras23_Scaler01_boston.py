from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
                                                

#1. DATA
dataset = load_boston()
x = dataset.data
y = dataset.target
# print(type(x))
# print(x)

# #정규화 변환
# print(np.min(x), np.max(x))     #0.0 711.0
# scaler = MinMaxScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# print(np.min(x), np.max(x))     #0.0 1.0
scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler = RobustScaler()

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=333
)

# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_test), np.max(x_test))     #-0.005578376185404939 1.1478180091225065

# #2. MODEL
# model = Sequential()
# model.add(Dense(1, input_dim=x.shape[1]))

# #3. COMPILE
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=10)

# #4. PREDICT
# loss = model.evaluate(x_test, y_test)
# print('loss: ', loss)