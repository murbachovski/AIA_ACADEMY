import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#1. DATA
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
             [9, 10, 11], [10, 11, 12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]]
             )
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x = x.reshape(13, 3, 1)

x_predict = np.array([50, 60, 70]) 

#2. MODEL
model = Sequential()
model.add(LSTM(10, input_shape=(3, 1), return_sequences=True))
model.add(LSTM(11, return_sequences=True))
model.add(GRU(11))
model.add(Dense(1))
model.summary()

