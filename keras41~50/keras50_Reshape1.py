from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM, GRU, Conv2D, MaxPooling2D, Conv1D, Reshape
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np


# [실습] 맹그러
# 목표는 cnn성능보다 좋게 맹그러!

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshape
x_train = x_train.reshape(-1, 28, 28, 1)/255.
x_test = x_test.reshape(-1, 28, 28, 1)/255.
# print(x_train.shape, y_train.shape)             # (60000, 28, 28, 1) (60000,)

#2. MODEL
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3, 3)))
model.add(Conv2D(10, 3,))
model.add(MaxPooling2D())
model.add(Flatten())    
model.add(Reshape(target_shape=(25, 10)))          
model.add(Conv1D(10, 3, padding='same'))
model.add(LSTM(784))
model.add(Reshape(target_shape=(28, 28, 1)))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

#  Layer (type)                Output Shape              Param #
# =================================================================
#  conv2d (Conv2D)             (None, 28, 28, 64)        640

#  max_pooling2d (MaxPooling2D  (None, 14, 14, 64)       0
#  )

#  conv2d_1 (Conv2D)           (None, 12, 12, 32)        18464

#  conv2d_2 (Conv2D)           (None, 10, 10, 10)        2890

#  max_pooling2d_1 (MaxPooling  (None, 5, 5, 10)         0
#  2D)

#  flatten (Flatten)           (None, 250)               0

#  reshape (Reshape)           (None, 25, 10)            0

#  conv1d (Conv1D)             (None, 25, 10)            310

#  lstm (LSTM)                 (None, 784)               2493120

#  reshape_1 (Reshape)         (None, 28, 28, 1)         0

#  conv2d_3 (Conv2D)           (None, 28, 28, 32)        320

#  flatten_1 (Flatten)         (None, 25088)             0

#  dense (Dense)               (None, 10)                250890

# =================================================================
# Total params: 2,766,634
# Trainable params: 2,766,634
# Non-trainable params: 0
# _________________________________________________________________