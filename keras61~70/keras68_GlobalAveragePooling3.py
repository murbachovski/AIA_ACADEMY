from keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D # 메모리 터질때 효율적이다.
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.random.set_seed(337)

# 1. DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
# (50000, 32, 32, 3) (50000, 1)
# (10000, 32, 32, 3) (10000, 1)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)

# Normalize data
# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. MODEL
model = Sequential()
model.add(Conv2D(64, (2, 2), padding='same', input_shape=(32, 32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(filters=64, kernel_size=(2, 2), padding='valid', activation='relu'))
model.add(Conv2D(33, 2))
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

import time
start = time.time()
model.fit(x_train, y_train, epochs=10, batch_size=128)
end = time.time()
print('Elapsed time:', end - start, 'seconds')

# 4. EVALUATE
results = model.evaluate(x_test, y_test)
print('Loss:', results[0])
print('Accuracy:', results[1])
