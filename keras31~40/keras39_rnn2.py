import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

#1. DATA
dataset = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# y = ?

x = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]]) 
y = np.array([5, 6, 7, 8, 9, 10])
print(x.shape, y.shape) #(6, 3) (7,)
#x의 shape =    (행, 열, 몇개씩 훈련하는지)
x = x.reshape(6, 4, 1) # => [[[1,]], [2], [3]], [[2], [3], [4]...........]]
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.025
)

#2. MODEL
model = Sequential()
model.add(SimpleRNN(32, input_shape=(4, 1)))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#3. VALUATE
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=100
)
model.fit(x, y, epochs=4000, validation_split=0.025, callbacks=[es], batch_size=2)
model.save('./_save/rnn_model.h5')

#4. PREDICT
loss = model.evaluate(x, y)

x_predict = np.array([7, 8, 9, 10]).reshape(1, 4, 1)
print(x_predict.shape)
result = model.predict(x_predict)
print('loss:', loss, 'result:', result)