import numpy as np
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, GRU, Bidirectional, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping

#1. DATA
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
             [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10],
             [9, 10, 11], [10, 11, 12],
             [20, 30, 40], [30, 40, 50], [40, 50, 60]]
             )
y = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 50, 60, 70])

x = x.reshape(13, 3, 1)

x_predict = np.array([50, 60, 70]) # I wanna 80

#실습

#2. MODEL
model = Sequential()
model.add(Conv1D(256, 2, input_shape=(3, 1), padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='loss',
    mode='auto',
    patience=30,
    restore_best_weights=True
)
model.fit(x, y, epochs=100, callbacks=[es], batch_size=8)

#4. PREDICT
loss = model.evaluate(x, y)
x_predict = x_predict.reshape(1, 3, 1)
result = model.predict(x_predict)
print('loss: ', loss, 'result: ', result)

