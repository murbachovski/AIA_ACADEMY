import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Flatten

model = Sequential()
# model.add(LSTM(10, input_shape=(3, 1)))       #total_params = 541
model.add(Conv1D(10, 2, input_shape=(3, 1)))    #total_params = 141
model.add(Conv1D(10, 2))                        #total_params = 301
model.add(Flatten())
model.add(Dense(5))
model.add(Dense(1))

model.summary()