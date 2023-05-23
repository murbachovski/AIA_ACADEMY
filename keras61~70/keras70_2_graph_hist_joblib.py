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


# Load the training history
# history = joblib.load('./keras70_1_mnist_graph.h5')


import joblib
try:
    hist = joblib.load('./keras70_1_history.dat')
except EOFError:
    print('###############EOFError 발생####################')

# Plot the training history
########################## 시각화 ##################### 
import matplotlib.pyplot as plt
plt.figure(figsize=(2,2))

plt.subplot(2,1,1)
plt.plot(hist['loss'], marker='.', c='red', label='loss')
plt.plot(hist['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(hist['acc'], marker='.', c='red', label='acc')
plt.plot(hist['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc', 'val_acc'])

plt.show()
