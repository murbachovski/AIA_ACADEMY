import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense, SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time

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
start = time.time()
#2. MODEL
model = Sequential()
model.add(SimpleRNN(64, input_shape=(5, 1)))
model.add(Dense(1))

#3. VALUATE
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='auto',
    restore_best_weights=True,
    patience=100
)
model.fit(x, y, epochs=1000, validation_split=0.025)

end = time.time()
#4. PREDICT
loss = model.evaluate(x, y)

x_predict = np.array([6, 7, 8, 9, 10]).reshape(1, 5, 1)
print(x_predict.shape)
result = model.predict(x_predict)
print('loss:', loss, 'result:', result)
print("걸린 시간: ", round(end - start, 2))