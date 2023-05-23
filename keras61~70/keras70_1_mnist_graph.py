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
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
model.add(Dense(10, activation='softmax'))

model.summary()

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
import time
start = time.time()
hist = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2)
end = time.time()
print('걸린 시간: ', end - start, 2)

# #4. EVALUATE
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc: ', results[1])

model.save('./keras70_1_mnist_graph.h5')

########################## 시각화 #####################
import matplotlib.pyplot as plt
plt.figure(figsize=(2,2))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()
