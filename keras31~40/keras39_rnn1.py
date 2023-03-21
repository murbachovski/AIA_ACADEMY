import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN

#1. DATA
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = ?

x = np.array([[1, 2, 3,], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]) 
y = np.array([4, 5, 6, 7, 8, 9, 10])
print(x.shape, y.shape) #(6, 3) (7,)
#x의 shape =    (행, 열, 몇개씩 훈련하는지)
x = x.reshape(7, 3, 1) # => [[[1,]], [2], [3]], [[2], [3], [4]...........]]
print(x.shape)

#2. MODEL
model = Sequential()
model.add(SimpleRNN(32, input_shape=(3, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

#3. VALUATE
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. PREDICT
loss = model.evaluate(x, y)

x_predict = np.array([8, 9, 10]).reshape(1, 3, 1)
print(x_predict.shape)
result = model.predict(x_predict)
