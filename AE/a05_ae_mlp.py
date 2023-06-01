import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import random

# 1. 데이터 로드 및 전처리
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 노이즈 추가
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# 2. 모델 구성
def autoencoder(hidden_layer_size):
    model = Sequential()
    # Encoder
    model.add(Dense(units=hidden_layer_size[0], input_shape=(784,), activation='relu'))
    model.add(Dense(units=hidden_layer_size[1], activation='relu'))
    # Bottleneck layer
    model.add(Dense(units=hidden_layer_size[2], activation='relu'))
    # Decoder
    model.add(Dense(units=hidden_layer_size[3], activation='relu'))
    model.add(Dense(units=hidden_layer_size[4], activation='relu'))
    model.add(Dense(784, activation='sigmoid'))  # Output layer with sigmoid activation
    
    return model

hidden_layer_size = [1000, 500, 100, 500, 1000]

model = autoencoder(hidden_layer_size=hidden_layer_size)

# 3. 모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=10, batch_size=128)

# 4. 이미지 예측 및 시각화
decoded_imgs = model.predict(x_test_noised)

fig, axes = plt.subplots(3, 5, figsize=(20, 7))

random_images = random.sample(range(decoded_imgs.shape[0]), 5)

for i, ax in enumerate(axes[0]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate(axes[1]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate(axes[2]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()
