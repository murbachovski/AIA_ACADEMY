import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#1. DATA
train_datagen = ImageDataGenerator(
    rescale=1./255,                     # MinMaxScaler, 정규화, 전처리, Nomalization
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1, 
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    rescale=1./255,                     # 평가 데이터이므로 증폭X
)
xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(100, 100),             # 각각 다른 사이즈를 가진 이미지들을 200, 200으로 맞춰준다. 
    batch_size=160,                       # 전체 데이터를 쓰려면 160(전체 데이터 갯수)이상을 넣어라
    class_mode='binary',
    color_mode='grayscale',
    # color_mode='rgba',
    shuffle=True
)                                       # Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100, 100),             # 각각 다른 사이즈를 가진 이미지들을 200, 200으로 맞춰준다.
    batch_size=200,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)                                       # Found 120 images belonging to 2 classes.

# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000270850C4F70>
# print(len(xy_train)) # 32
# print(len(xy_train[0])) # 2
# print(xy_train[0][0])   # x 다섯 개 들어가 있다.
# print(xy_train[0][1])   # [1. 0. 0. 0. 1.]
print(xy_train[0][0].shape)   
print(xy_train[0][1].shape)

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
model.add(Dense(1, activation='sigmoid'))

#3. COMPILE
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(xy_train[:][0], xy_train[:][1], epochs=10) # error
# model.fit(xy_train[0][0], xy_train[0][1], epochs=10)   # 통배치 넣으면 가능하다.
Es = EarlyStopping(
    monitor='val_acc',
    patience=20,
    mode='max',
    restore_best_weights=True
)
hist = model.fit(xy_train[0][0],
        xy_train[0][1],
        batch_size=26,
        epochs=10,
        steps_per_epoch=1,
        validation_data=(xy_test[0][0], xy_test[0][1])
        )
# hist = model.fit_generator(xy_train, epochs=200,
#                     steps_per_epoch=1, # 전체 데이터/batch = 160/5 = 160/5 = 32
#                     validation_data=xy_test,
#                     # validation_steps=24 # validation_data/batch = 120/5 = 24 
#                     )
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
import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 6)) 
# plt.plot(hist.history['loss'], marker = '.', c='red')
# plt.plot(hist.history['val_loss'], marker = '.', c='blue')
# plt.plot(hist.history['acc'], marker = '.', c='green')
# plt.plot(hist.history['val_acc'], marker = '.', c='black')
# plt.show()

# plt.subplot(2, 1, 1)                # nrows=2, ncols=1, index=1
# plt.plot(val_loss, 'o-')
# plt.title('loss')

# plt.subplot(2, 1, 2)                # nrows=2, ncols=1, index=2
# plt.plot(val_acc, 'o-')
# plt.title('acc')

# plt.tight_layout()
# plt.show()
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(111)
ax1 = plt.subplot(2,2,1)   # 4x4의 첫 번째
ax2 = plt.subplot(2,2,2)   # 4x4의 두 번째
ax1.plot(loss)   # 첫 번째에 그림을 그린다.
ax1.plot(val_loss)   # 첫 번째에 그림을 추가한다.
ax2.plot(acc)   # 네 번째에 그림을 그린다.
ax2.plot(val_acc)   # 네 번째에 그림을 그린다.
plt.show()