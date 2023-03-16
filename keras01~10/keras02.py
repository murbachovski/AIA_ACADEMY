#23.02.28

#1.data
#numpy = 수치화 활용
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
# loss = 0에 수렴 = 최소의loss


#2.model
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3.compile
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)
#fit = 훈련을 시키다.
#epochs = 반복 횟수   
#loss = 0.0058