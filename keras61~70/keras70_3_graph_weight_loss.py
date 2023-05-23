from keras.datasets import mnist
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib

tf.random.set_seed(337)

# 1. DATA
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# ONE_HOT
y_train = to_categorical(y_train) 
y_test = to_categorical(y_test)

# Load the model
model = load_model('model.h5')

# Load the training history
history = joblib.load('history.joblib')

# 3. COMPILE
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# # 4. TRAIN
# hist = model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.2)

# Save the training history
loss = history['loss']
val_loss = history['val_loss']
acc = history['acc']
val_acc = history['val_acc']

# Plot the training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], marker='.', c='red', label='loss')
plt.plot(history['val_loss'], marker='.', c='blue', label='val_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['acc'], marker='.', c='red', label='acc')
plt.plot(history['val_acc'], marker='.', c='blue', label='val_acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Visualize the model's weights
fig, axs = plt.subplots(len(model.layers), figsize=(8, 6))

for i, layer in enumerate(model.layers):
    if hasattr(layer, 'weights'):
        weights = layer.get_weights()
        if weights:
            weight_values = weights[0]
            axs[i].hist(weight_values.flatten(), bins=30)
            axs[i].set_title(f"Weights for {layer.name}")
            axs[i].set_xlabel('Weight Value')
            axs[i].set_ylabel('Frequency')

plt.tight_layout()
plt.show()