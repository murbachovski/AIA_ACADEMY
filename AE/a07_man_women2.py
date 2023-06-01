import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D
from matplotlib import pyplot as plt
import random
import tensorflow as tf

path = './_data/'
save_path = 'd:/_save/'

all_data = ImageDataGenerator(
    rescale=1./255
)

start = time.time()

people_data = all_data.flow_from_directory(
    path,
    target_size=(100, 100),
    batch_size=100,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

x_data = people_data[0][0]
y_data = people_data[0][1]

end = time.time()

print(end - start)

# Add noise
x_data_noised = x_data + np.random.normal(0, 0.2, size=x_data.shape)
x_data_noised = np.clip(x_data_noised, a_min=0, a_max=1)

# Model configuration
model = Sequential()
# Encode
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(100, 100, 3)))
model.add(MaxPool2D())
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

# Decode
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D())
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

model.summary()

# Compile and train the model
model.compile(optimizer='adam', loss='mse')
model.fit(x_data_noised, x_data, epochs=100, batch_size=64)

# Select one image
index = 0

# Add noise to the selected image
noisy_image = x_data[index] + np.random.normal(0, 0.1, size=x_data[index].shape)
noisy_image = np.clip(noisy_image, a_min=0, a_max=1)

# Reshape the noisy image for prediction
noisy_image = np.expand_dims(noisy_image, axis=0)

# Generate the denoised image
denoised_image = model.predict(noisy_image)

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(x_data[index].reshape(100, 100, 3), cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(noisy_image.reshape(100, 100, 3), cmap='gray')
axes[1].set_title('Noisy')
axes[1].axis('off')

axes[2].imshow(x_data[index].reshape(100, 100, 3), cmap='pink_r')
axes[2].set_title('Denoised')
axes[2].axis('off')


plt.show()
