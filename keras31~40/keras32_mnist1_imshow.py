import numpy as np
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)     #(60000, 28, 28) (60000,) == 흑백
print(x_test.shape, y_test.shape)       #(10000, 28, 28) (10000,)
#reshape해줘야해요.
print(x_train)
print(y_train)      #[5 0 4 ... 5 6 8]
print(x_train[0])
print(y_train[0])   #5
import matplotlib.pyplot as  plt
plt.imshow(x_train[0], 'Blues')
plt.show()