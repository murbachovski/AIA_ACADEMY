#불러와서 모델 완성
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
import time
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
path = 'd:/study_data/_data/rps/'
save_path = 'd:/study_data/_save/rps/'

x_train = np.load(save_path + 'keras56_rps_x_train.npy')
x_test = np.load(save_path + 'keras56_rps_x_test.npy')
y_train = np.load(save_path + 'keras56_rps_y_train.npy')
y_test = np.load(save_path + 'keras56_rps_y_test.npy')


#2. MODEL
model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. COMPILE
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
Es = EarlyStopping(
    monitor='val_acc',
    patience=200,
    mode='auto',
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test), callbacks=[Es])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:', val_acc[-1])

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

loss: 9.868523420664133e-07
val_loss: 0.1739882081747055
acc: 1.0
val_acc: 1.0