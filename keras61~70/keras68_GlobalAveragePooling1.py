from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import tensorflow as tf
tf.random.set_seed(337)

#1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1) #내용과 순서는 바뀌면 안된다.
x_test = x_test.reshape(10000, 28, 28, 1)   #구조만 바뀌는 것이다.
                                            #(60000<데이터 갯수>, 28, 28, 1)

# print(np.unique(y_train, return_counts=True)) #(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949], dtype=int64))

#실습

# ONE_HOT
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

print(y_train)
print(y_train.shape) #(60000, 10)

#2. MODEL
model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D()) #중첩하지 않게 제일 큰 놈을 가져온다. 기본 값은 (2,2)
model.add(Conv2D(filters=64, kernel_size=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(32, 2)) # 2 => (2,2) 귀찮아서 줄여서 쓸 수 있다.
model.add(Flatten())
# model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.summary()
#################################### Flatten 연산량 ###########################
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        320
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        16448
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 12, 12, 32)        8224
# _________________________________________________________________
# flatten (Flatten)            (None, 4608)              0
# _________________________________________________________________
# dense (Dense)                (None, 10)                46090
# =================================================================
# Total params: 71,082

#################################### GlobalAveragePooling2D 연산량 ###########################
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 28, 28, 64)        320
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 14, 14, 64)        0
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 13, 13, 64)        16448
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 12, 12, 32)        8224
# _________________________________________________________________
# module_wrapper (ModuleWrappe (None, 32)                0
# _________________________________________________________________
# dense (Dense)                (None, 10)                330
# =================================================================
# Total params: 25,322

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
import time
start = time.time()
model.fit(x_train, y_train, epochs=30, batch_size=128)
end = time.time()
print('걸린 시간: ', end - start, 2)

# 걸린 시간:  91.4720869064331 2
# 313/313 [==============================] - 1s 3ms/step - loss: 0.1863 - acc: 0.9828 
# loss:  0.18631193041801453 acc:  0.9828000068664551

# 걸린 시간:  96.60862708091736 2
# 313/313 [==============================] - 1s 3ms/step - loss: 0.1791 - acc: 0.9444
# loss:  0.17905108630657196 acc:  0.9444000124931335

# #4. EVALUATE
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc: ', results[1])

# loss:  1.6532992124557495 acc:  0.9839000105857849

# # reshape #2차원으로 바꿔주기.
# x_train = x_train.reshape(60000, 28*28)
# x_test = x_test.reshape(10000, 28*28)

# # SCALER #2차원만 가능합니다.
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

# #reshape
# x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)

# #2. MODEL
# model = Sequential()
# model.add(Conv2D(10,(2,2), input_shape=(28, 28, 1)))
# model.add(Conv2D(filters=4, kernel_size=(3,3), activation='relu'))
# model.add(Conv2D(2, (2,2)))                                                                                   
# model.add(Conv2D(2, (2,2)))                                                                                   
# model.add(Conv2D(2, (2,2)))                                                                                   
# model.add(Conv2D(2, (2,2)))                                                                                   
# model.add(Flatten())
# model.add(Dense(10))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax'))
# model.summary()

# #3. COMPILE
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs=50, batch_size=1000)

# #4. EVALUATE
# results = model.evaluate(x_test, y_test)
# print('loss: ', results[0], 'acc: ', results[1])

