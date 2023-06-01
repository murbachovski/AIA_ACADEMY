import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from matplotlib import pyplot as plt
import random

# 1. 데이터 로드 및 전처리
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(10000, 28, 28, 1).astype('float32') / 255

# 노이즈 추가
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

# 2. 모델 구성
model = Sequential()
# Encode
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

# Decode
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

model.summary()

# 3. 모델 컴파일 및 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train_noised, x_train, epochs=1, batch_size=128)

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
