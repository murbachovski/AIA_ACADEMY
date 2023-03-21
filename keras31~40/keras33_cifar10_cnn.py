from keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import time
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, mean_squared_error

#1. DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)     #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)       #(10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000], dtype=int64))

# ONE_HOT
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# SCALER MinMaxScaler() == x_train = x_train/255. x_test = x_test/255.
x_train = x_train/255.
x_test = x_test/255.

#2. MODEL
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(32, 32, 3)))
model.add(Conv2D(filters=4, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(2, (2,2)))
model.add(MaxPooling2D())                                                                             
model.add(Conv2D(2, (2,2)))                                                                                   
model.add(Flatten())
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.summary()

es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    restore_best_weights=True,
    patience=10
)

start_time = time.time()
#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=200, validation_split=0.3, callbacks=[es])
end_time = time.time()
print('걸린시간: ', round(end_time - start_time, 2))

#4. EVALUATE
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
def RMSE(y_test, y_predict):
    return mean_squared_error(y_test, y_predict)
rmse = RMSE(y_test, y_predict)

print('loss: ', results[0], 'acc: ', results[1], 'r2: ', r2, 'rmse: ', rmse)
