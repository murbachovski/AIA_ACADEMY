import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#1. DATA
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = ?

x = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]) 
y = np.array([6, 7, 8, 9, 10])
print(x.shape, y.shape) #(6, 3) (7,)
#x의 shape =    (행, 열, 몇개씩 훈련하는지)
x = x.reshape(5, 5, 1) # => [[[1,]], [2], [3]], [[2], [3], [4]...........]]
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.025
)

#2. MODEL
model = Sequential()
model.add(LSTM(10, input_shape=(5, 1)))
model.add(Dense(7))
model.add(Dense(1))
model.summary()

# loss:  7.076236215652898e-05 result:  [[78.20666]]