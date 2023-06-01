import numpy as np
from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 잡음을 넣는다.
x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


#2. MODEL
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input
# 함수형..!
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,)))  # Fixed typo here
    model.add(Dense(784, activation='relu'))    
    return model

model1 = autoencoder(hidden_layer_size=1)
model32 = autoencoder(hidden_layer_size=32)
model64 = autoencoder(hidden_layer_size=64)
model154 = autoencoder(hidden_layer_size=154)
model331 = autoencoder(hidden_layer_size=331)
# model486 = autoencoder(hidden_layer_size=486)
# model713 = autoencoder(hidden_layer_size=713)

# 컴파일 훈련
model1.compile(optimizer='adam', loss='mse')
model1.fit(x_train, x_train, epochs=10, batch_size=128)  # Fixed typo here
# 컴파일 훈련
model32.compile(optimizer='adam', loss='mse')
model32.fit(x_train, x_train, epochs=10, batch_size=128) 
# 컴파일 훈련
model64.compile(optimizer='adam', loss='mse')
model64.fit(x_train, x_train, epochs=10, batch_size=128) 
# 컴파일 훈련
model154.compile(optimizer='adam', loss='mse')
model154.fit(x_train, x_train, epochs=10, batch_size=128) 
# 컴파일 훈련
model331.compile(optimizer='adam', loss='mse')
model331.fit(x_train, x_train, epochs=10, batch_size=128) 

# 평가, 예측
decoded_imgs1 = model1.predict(x_test_noised)
# 평가, 예측
decoded_imgs32 = model32.predict(x_test_noised)
# 평가, 예측
decoded_imgs64 = model64.predict(x_test_noised)
# 평가, 예측
decoded_imgs154 = model154.predict(x_test_noised)
# 평가, 예측
decoded_imgs331 = model331.predict(x_test_noised)

######################################################################################

from matplotlib import pyplot as plt
import random

fig, axes = plt.subplots(7, 5, figsize=(15, 15))  # Fixed typo here

random_images = random.sample(range(decoded_imgs1.shape[0]), 5)
outputs = [x_test, decoded_imgs1, decoded_imgs32, decoded_imgs64, decoded_imgs154, decoded_imgs331]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(x_test[random_images[row_num]].reshape(28, 28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
