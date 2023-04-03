from tensorflow.keras.datasets import mnist
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

path = 'd:/temp/'
save_path = 'd:/temp/'
stt = time.time()
#1. 데이터 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

np.random.seed(1111)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

train_datagen2 = ImageDataGenerator(
    rescale=1./1,
)


train_datagen2 = ImageDataGenerator(
    rescale=1./1,
)

augment_size = 40000
batch_size=64
randidx=np.random.randint(x_train.shape[0], size=augment_size)

x_augmented = x_train[randidx].copy()      
y_augmented = y_train[randidx].copy() 

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 
                        x_test.shape[1], 
                        x_test.shape[2],
                        1)

x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2],
                                  1)

x_augmented = train_datagen.flow(
    x_augmented, y_augmented, 
    batch_size = augment_size,
    shuffle=False,
).next()[0] 

x_train = np.concatenate((x_train/255., x_augmented))   
y_train = np.concatenate((y_train, y_augmented))   
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
x_test = x_test/255.

xy_train = train_datagen2.flow(x_train, y_train,
                               batch_size = batch_size, shuffle=True)

np.save(save_path + 'keras58_mnist_x_train.npy', arr=x_train)
np.save(save_path + 'keras58_mnist_x_test.npy', arr=x_test)
np.save(save_path + 'keras58_mnist_y_train.npy', arr=y_train)
np.save(save_path + 'keras58_mnist_y_test.npy', arr=y_test)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])

# model.fit(xy_train[:][0], xy_train[:][1],
#           epochs=10,
#           )   #에러

es = EarlyStopping(monitor='val_acc',
                   mode = 'max',
                   patience=30,
                   verbose=1,
                   restore_best_weights=True,
                   )

hist = model.fit_generator(xy_train, epochs=10,   #x데이터 y데이터 배치사이즈가 한 데이터에 있을때 fit 하는 방법
                    steps_per_epoch=len(xy_train)/batch_size,    #전체데이터크기/batch = 160/5 = 32
                    # validation_split=0.1,
                    shuffle=True,
                    # batch_size = 16,
                    # validation_steps=24,  
                    callbacks=[es],
                    # validation_data=[x_test, y_test]
                    )


loss = hist.history['loss']
# val_loss = hist.history['val_loss']
acc = hist.history['acc']
# val_acc = hist.history['val_acc']

# print('loss : ', loss[-1])
# print('val_loss : ', val_loss[-1])
# print('acc : ', acc[-1])
# print('val_acc : ', val_acc[-1])

np.save(save_path + 'keras58_mnist_x_train.npy', arr=x_train)
np.save(save_path + 'keras58_mnist_x_test.npy', arr=x_test)
np.save(save_path + 'keras58_mnist_y_train.npy', arr=y_train)  
np.save(save_path + 'keras58_mnist_y_test.npy', arr=y_test)

ett1 = time.time()
print('이미지 증폭 시간 :', np.round(ett1-stt, 2))
