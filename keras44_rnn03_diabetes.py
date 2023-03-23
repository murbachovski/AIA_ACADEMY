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
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, GRU
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
                                                

#1. DATA
dataset = load_diabetes()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2, 
    random_state=333
)

print(x_train.shape, x_test.shape) #(353, 10) (89, 10)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_test), np.max(x_test))

x_train = x_train.reshape(353, 10, 1)
x_test = x_test.reshape(89, 10, 1)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_shape=(10, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs=10)

#4. PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)