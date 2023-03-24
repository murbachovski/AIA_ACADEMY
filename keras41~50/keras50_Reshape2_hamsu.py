from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, LSTM, GRU, Conv2D, MaxPooling2D, Conv1D, Reshape, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import numpy as np

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
# _________________________________________________________________
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

# _________________________________________________________________# _________________________________________________________________# _________________________________________________________________

# input1 = Input(shape=(28, 28, 1))
# conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(input1)
# mp1 = MaxPooling2D()(conv1)
# conv2 = Conv2D(32, 3)(mp1)
# conv3 = Conv2D(10, 3)(conv2)
# mp2 = MaxPooling2D()(conv3)
# flat1 = Flatten()(mp2)
# rs1 = Reshape(target_shape=(25, 10))(flat1)
# conv4 = Conv1D(10, 3, padding='same')(rs1)
# lstm1 = LSTM(784)(conv4)
# rs2 = Reshape(target_shape=(28, 28, 1))(lstm1)
# conv5 = Conv2D(32, 3, padding='same')(rs2)
# flat2 = Flatten()(conv5)
# output1 = Dense(10, activation='softmax')(flat2)
# model = Model(inputs=input1, outputs=output1)
# model.summary()


# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  input_1 (InputLayer)        [(None, 28, 28, 1)]       0

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