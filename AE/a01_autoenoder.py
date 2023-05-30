# 잡음 제거
# 기미, 주근깨 제거
# 이미지 데이터에 많이 씀
# 하지만 모든 데이터에 사용 가능
# 준지도 학습
# x로 x를 훈련시킨다. (y값은 필요가 없다)
# 원본 사진 두장을 그대로 훈련이 가능하다

import numpy as np
from tensorflow.keras.datasets import mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 라벨 값이 필요가 없기에 
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000, 784).astype('float32') / 255.
x_test = x_test.reshape(10000, 784).astype('float32') / 255.

# from keras.models import Sequential # Another Version
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import Dense, Input
# 함수형..!

input_img = Input(shape=(784,))
encoded = Dense(64, activation='relu')(input_img)
# decoded = Dense(784 ,activation='sigmoid')(encoded) # 첫번째  # x 자체
# 375/375 [==============================] - 2s 6ms/step - loss: 0.0729 - acc: 0.0130 - val_loss: 0.0738 - val_acc: 0.0134
decoded = Dense(784 ,activation='relu')(encoded) # 두번째    # x 자체
autoencoder = Model(input_img, decoded)
# 64 => 784 => 64 => 784 반복
autoencoder.summary()

#       #
#   #   #
#   #   #
#       #
# 특성을 추출하다가 경계선이 흐릿해지는 현상이 발생할 수 있다.

autoencoder.compile(optimizer = 'adam', loss = 'mse', metrics = ['acc'])
# autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

autoencoder.fit(x_train, x_train, epochs= 30, batch_size = 128, validation_split=0.2) # x로 x를 훈련
