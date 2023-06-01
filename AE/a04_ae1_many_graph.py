import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import random

# 1. 데이터 불러오기
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 잡음을 넣는다.
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)


# 2. 모델 정의
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,)))
    model.add(Dense(784, activation='relu'))
    return model

models = [
    autoencoder(hidden_layer_size=1),
    autoencoder(hidden_layer_size=32),
    autoencoder(hidden_layer_size=64),
    autoencoder(hidden_layer_size=154),
    autoencoder(hidden_layer_size=331)
]

# 3. 모델 학습 및 결과 확인
decoded_imgs_all = []

for model in models:
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train_noised, x_train_noised, epochs=3, batch_size=128)

    decoded_imgs = model.predict(x_test_noised)
    decoded_imgs_all.append(decoded_imgs)

fig, axes = plt.subplots(len(decoded_imgs_all), 5, figsize=(15, 15))

for i, decoded_imgs in enumerate(decoded_imgs_all):
    random_images = random.sample(range(decoded_imgs.shape[0]), 5)
    outputs = [x_test_noised, decoded_imgs]

    for j, ax in enumerate(axes[i]):
        ax.imshow(outputs[j][random_images[j]].reshape(28, 28), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
