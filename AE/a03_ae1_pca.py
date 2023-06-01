import numpy as np
from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 잡음을 넣는다.
x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape)

# print(x_train_noised.shape, x_test_noised.shape) # (60000, 784) (10000, 784)
# print(np.max(x_train_noised), np.min(x_train_noised))
# print(np.max(x_test_noised), np.min(x_test_noised))
# 1.502660591712122 -0.5356823766615968
# 1.5003927198446805 -0.5715293818320267
# 0 ~ 1 사이 값으로 바꿔주기!
#

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
# print(np.max(x_train_noised), np.min(x_train_noised))
# print(np.max(x_test_noised), np.min(x_test_noised))
# 1.0 0.0
# 1.0 0.0


#2. MODEL
# from keras.models import Sequential # Another Version
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input
# 함수형..!
def autoencoder(hidden_layer_size,):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,)))
    model.add(Dense(784, activation='relu'))    
    return model

# model = autoencoder(hidden_layer_size=154)
# model = autoencoder(hidden_layer_size=331)
# model = autoencoder(hidden_layer_size=486)
model = autoencoder(hidden_layer_size=713)



# 컴파일 훈련
model.compile(optimizer = 'adam', loss= 'mse')
model.fit(x_train, x_train, epochs=10, batch_size=128)

# 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n , i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n , i+1+n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()