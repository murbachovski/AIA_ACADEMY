import numpy as np
from tensorflow.keras.datasets import mnist

#1. DATA
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# 잡음을 넣는다.
x_train_noised = x_train + np.random.normal(0, 0.1, size= x_train.shape)
x_test_noised = x_test + np.random.normal(0, 0.1, size= x_test.shape)

# print(x_train_noised.shape, x_test_noised.shape) # (60000, 784) (10000, 784)
# print(np.max(x_train_noised), np.min(x_train_noised))
# print(np.max(x_test_noised), np.min(x_test_noised))
# 1.502660591712122 -0.5356823766615968
# 1.5003927198446805 -0.5715293818320267
# 0 ~ 1 사이 값으로 바꿔주기!
#

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
# print(np.max(x_train_noised), np.min(x_train_noised))
# print(np.max(x_test_noised), np.min(x_test_noised))
# 1.0 0.0
# 1.0 0.0


#2. MODEL
# from keras.models import Sequential # Another Version
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input
# 함수형..!

input_img = Input(shape=(784,))
encoded = Dense(1024, activation='relu')(input_img)
decoded = Dense(784 ,activation='relu')(encoded) # 첫번째  # x 자체
autoencoder = Model(input_img, decoded)
autoencoder.summary()
#3. COMPILE, FIT
autoencoder.compile(optimizer = 'adam', loss = 'mse')
# autoencoder = sigmoid / mse 조합이 잘 먹힌다.
autoencoder.fit(x_train_noised, x_train_noised, epochs= 1, batch_size = 128) # x로 x를 훈련
# 앞뒤가 똑같은...
#4. 평가, 예측
decoded_imgs = autoencoder.predict(x_test_noised)



import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n , i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    ax = plt.subplot(2, n , i+1+n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
# 469/469 [==============================] - 3s 4ms/step - loss: 0.0127