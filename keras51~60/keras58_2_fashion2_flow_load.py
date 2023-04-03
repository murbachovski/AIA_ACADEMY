from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score 
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from tensorflow.keras.utils import to_categorical

path = 'd:/temp/'
save_path = 'd:/temp/'


#1. 데이터 
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) 
# print(x_test.shape, y_test.shape)

np.random.seed(1111)

x_train = np.load(save_path + 'keras58_fashion_x_train.npy')
x_test = np.load(save_path + 'keras58_fashion_x_test.npy')
y_train = np.load(save_path + 'keras58_fashion_y_train.npy')
y_test = np.load(save_path + 'keras58_fashion_y_test.npy')


#2. MODEL
model = Sequential()
model.add(Conv2D(256, (2,2), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
Es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=100,
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[Es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']
print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:', val_acc[-1])

# loss: 0.6086599230766296
# val_loss: 1.832023024559021
# acc: 0.7599999904632568
# val_acc: 0.4399999976158142

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.subplot(1, 2, 1)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.grid()
plt.legend()
plt.show()