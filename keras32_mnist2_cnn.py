from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, RobustScaler


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

# reshape #2차원으로 바꿔주기.
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

# SCALER #2차원만 가능합니다.
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#reshape
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

#2. MODEL
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(28, 28, 1)))
model.add(Conv2D(filters=4, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(2, (2,2)))                                                                                   
model.add(Conv2D(2, (2,2)))                                                                                   
model.add(Conv2D(2, (2,2)))                                                                                   
model.add(Conv2D(2, (2,2)))                                                                                   
model.add(Flatten())
model.add(Dense(10))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))
model.summary()

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=50, batch_size=1000)

#4. EVALUATE
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc: ', results[1])

