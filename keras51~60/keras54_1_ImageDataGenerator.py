import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255, #MinMaxScaler, 정규화, 전처리, Nomalization
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
    rescale=1./255, # 평가 데이터이므로 증폭X
)
xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/brain/train/',
    target_size=(200, 200),     # 각각 다른 사이즈를 가진 이미지들을 200, 200으로 맞춰준다. 
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)                               # Found 160 images belonging to 2 classes.
xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/brain/test/',
    target_size=(200, 200),     # 각각 다른 사이즈를 가진 이미지들을 200, 200으로 맞춰준다.
    batch_size=5,
    class_mode='binary',
    color_mode='grayscale',
    shuffle=True
)                                # Found 120 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000270850C4F70>
print(len(xy_train)) # 32
print(len(xy_train[0])) # 2
print(xy_train[0][0])   # x 다섯 개 들어가 있다.
print(xy_train[0][1])   # [1. 0. 0. 0. 1.]
print(xy_train[0][0].shape)   
print(xy_train[0][1].shape)

print('###############################################################################')
print(type(xy_train))       # <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple' = 바꿀 수 없다.> (첫번 째 batch)
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>

