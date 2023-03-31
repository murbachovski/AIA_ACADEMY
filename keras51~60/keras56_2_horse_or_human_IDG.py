import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split
import time
from keras.callbacks import EarlyStopping

path = 'd:/study_data/_data/horse_human/'
save_path = 'd:/study_data/_save/horse_human/'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

all_data = ImageDataGenerator(
    rescale=1./255
)

start = time.time()

h_h_data = all_data.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

h_h_x = h_h_data[0][0]
h_h_y = h_h_data[0][1]

end = time.time()
print(end - start)

hhx_train, hhx_test, hhy_train, hhy_test = train_test_split(
    h_h_x,
    h_h_y,
    test_size=0.025,
    shuffle=True,
    random_state=2222
)

# print(all_data)
# print(h_h_data)
# print(h_h_x)
# print(h_h_y)
# print(h_h_x.shape) # (100, 100, 100, 3)
# print(h_h_y.shape) # (100,)

np.save(save_path + 'keras56_horse_human_x_train.npy', arr=hhx_train)
np.save(save_path + 'keras56_horse_human_x_test.npy', arr=hhx_test)
np.save(save_path + 'keras56_horse_human_y_train.npy', arr=hhy_train)
np.save(save_path + 'keras56_horse_human_y_test.npy', arr=hhy_test)
print(hhx_train.shape, hhx_test.shape) # (97, 100, 100, 3) (3, 100, 100, 3)
print(hhy_train.shape, hhy_test.shape) # (97,) (3,)

#2. MODEL
model = Sequential()
model.add(Conv2D(128, (2, 2), input_shape=(100, 100, 3), activation='relu'))
model.add(Conv2D(64, (2, 2), activation='relu'))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. COMPILE
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
Es = EarlyStopping(
    monitor='val_acc',
    patience=50,
    mode='auto',
    restore_best_weights=True
)
hist = model.fit(hhx_train, hhy_train, epochs=100, validation_data=(hhx_test, hhy_test), callbacks=[Es])

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

# loss: 1.623747380108398e-06
# val_loss: 1.2922411087856744e-06
# acc: 1.0
# val_acc: 1.0