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
model.add(LSTM(128, input_shape=(3, 1), return_sequences=True))
model.add(LSTM(64, return_sequences=True, activation='relu'))
model.add(GRU(32, return_sequences=True, activation='relu'))
model.add(GRU(16, return_sequences=True, activation='relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='loss',
    mode='auto',
    patience=100,
    restore_best_weights=True
)
model.fit(x, y, epochs=1000, callbacks=[es], batch_size=30)

#4. PREDICT
loss = model.evaluate(x, y)
x_predict = x_predict.reshape(1, 3, 1)
result = model.predict(x_predict)
print('loss: ', loss, 'result: ', result)
#loss:  425.6966247558594 result:  [[17.835804]]
#loss:  355.49560546875 result:  [[21.609966]]
#loss:  751.7981567382812 result:  [[[4.245128 ]
#loss:  662.6885375976562 result:  [[[45.44535 ]