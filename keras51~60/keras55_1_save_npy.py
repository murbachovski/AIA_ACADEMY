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
    batch_size=5000,                       # 전체 데이터를 쓰려면 160(전체 데이터 갯수)이상을 넣어라
    class_mode='binary',
    color_mode='grayscale',
    # color_mode='rgba',
    shuffle=True
)                                       # Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(100, 100),             # 각각 다른 사이즈를 가진 이미지들을 200, 200으로 맞춰준다.
    batch_size=120,                      
    class_mode='binary',                # y의 클래스
    # 0, 1 => 1 0 , 0 1로 원핫인코딩 됨
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
print(xy_test[0][0].shape)   
print(xy_test[0][1].shape)

path = 'd:/study_data/_save/_npy/'
np.save(path + 'keras55_1_x_train.npy', arr=xy_train[0][0])
np.save(path + 'keras55_1_x_test.npy', arr=xy_test[0][0])
np.save(path + 'keras55_1_y_train.npy', arr=xy_train[0][1])
np.save(path + 'keras55_1_y_test.npy', arr=xy_test[0][1])


'''
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
hist = model.fit(xy_train,
                epochs=50,
                steps_per_epoch=10, # 전체 데이터/batch = 160/5 = 160/5 = 32
                validation_data=xy_test,
                validation_steps=10 # validation_data/batch = 120/5 = 24 
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
'''