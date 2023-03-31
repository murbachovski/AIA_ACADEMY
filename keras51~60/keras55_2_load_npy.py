import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. DATA
path = 'd:/study_data/_save/_npy/'
# np.save(path + 'keras55_1_x_train.npy', arr=xy_train[0][0])
# np.save(path + 'keras55_1_x_test.npy', arr=xy_test[0][0])
# np.save(path + 'keras55_1_y_train.npy', arr=xy_train[0][1])
# np.save(path + 'keras55_1_y_test.npy', arr=xy_test[0][1])

x_train = np.load(path + 'keras55_1_x_train.npy')
x_test = np.load(path + 'keras55_1_x_test.npy')
y_train = np.load(path + 'keras55_1_y_train.npy')
y_test = np.load(path + 'keras55_1_y_test.npy')
# print(x_train)
# print(x_train.shape, x_test.shape) # (160, 100, 100, 1) (120, 100, 100, 1)
# print(y_train.shape, y_test.shape) # (160,) (120,)


# print('###############################################################################')
# # print(type(xy_train))                 # <class 'keras.preprocessing.image.DirectoryIterator'>
# # print(type(xy_train[0]))              # <class 'tuple' = 바꿀 수 없다.> (첫번 째 batch)
# # print(type(xy_train[0][0]))           # <class 'numpy.ndarray'>
# # print(type(xy_train[0][1]))           # <class 'numpy.ndarray'>
#현재 x는 (5, 200, 200, 1) 데이터가 32덩어리

#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(64, (2, 2), input_shape=(100, 100, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. COMPILE
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(xy_train[:][0], xy_train[:][1], epochs=10) # error
# model.fit(xy_train[0][0], xy_train[0][1], epochs=10)   # 통배치 넣으면 가능하다.
Es = EarlyStopping(
    monitor='val_acc',
    patience=20,
    mode='max',
    restore_best_weights=True
)
hist = model.fit(x_train,
                y_train,
                epochs=50,
                )


loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['acc']
val_acc = hist.history['val_acc']

print('loss:', loss[-1])
print('val_loss:', val_loss[-1])
print('acc:', acc[-1])
print('val_acc:', val_acc[-1])

#1. 그림그려
#2. 튜닝 0.95이상
from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.plot(range(len(hist.history['loss'])),hist.history['loss'],label='loss')
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(hist.history['acc'])),hist.history['acc'],label='acc')
plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],label='val_acc')
plt.legend()
plt.show()