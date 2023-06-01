import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from matplotlib import pyplot as plt
import random

#1. DATA
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Add noise
x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)

#2. MODEL
def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, input_shape=(784,)))
    model.add(Dense(784, activation='relu'))    
    return model

hidden_layer_sizes = [1, 32, 64, 154, 331]
models = []
decoded_imgs = []

for hidden_size in hidden_layer_sizes:
    model = autoencoder(hidden_layer_size=hidden_size)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, x_train, epochs=2, batch_size=128)
    models.append(model)
    decoded_imgs.append(model.predict(x_test_noised))

fig, axes = plt.subplots(len(hidden_layer_sizes) + 1, 5, figsize=(15, 15))

random_images = random.sample(range(x_test.shape[0]), 5)
outputs = [x_test] + decoded_imgs

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        if row_num == 0:
            ax.imshow(outputs[row_num][random_images[col_num]].reshape(28, 28), cmap='gray')
        else:
            ax.imshow(outputs[row_num][random_images[col_num]].reshape(28, 28), cmap='gray')  # Reshape the image
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()

