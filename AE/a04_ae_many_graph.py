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

# model = autoencoder(hidden_layer_size=154)
# model = autoencoder(hidden_layer_size=331)
# model = autoencoder(hidden_layer_size=486)
model = autoencoder(hidden_layer_size=713)

# 컴파일 훈련
model.compile(optimizer='adam', loss='mse')
model.fit(x_train, x_train, epochs=10, batch_size=128)  # Fixed typo here

# 평가, 예측
decoded_imgs = model.predict(x_test_noised)

######################################################################################

from matplotlib import pyplot as plt
import random
fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10),
      (ax11, ax12, ax13, ax14, ax15)) = \
      plt.subplots(3, 5, figsize=(20, 7))  # Fixed typo here

random_images = random.sample(range(decoded_imgs.shape[0]), 5)

for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28, 28), cmap='gray')
    if i == 0:
        ax.set_ylabel("INPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(x_test_noised[random_images[i]].reshape(28, 28), cmap='gray')  # Show noisy images
    if i == 0:
        ax.set_ylabel("NOISE", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow(decoded_imgs[random_images[i]].reshape(28, 28), cmap='gray')  # Show reconstructed images
    if i == 0:
        ax.set_ylabel("OUTPUT", size=20)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()
