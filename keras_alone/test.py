from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(10,(2,2),input_shape=(9, 8, 1)))
model.add(Conv2D(filters=3, kernel_size=(3,3)))
model.add(Conv2D(5, (3,3), activation='relu'))
model.add(Conv2D(3, (2,2)))
model.add(Conv2D(2, (2,2)))
model.add(Flatten())
model.add(Dense(32))
model.add(Dense(3, activation='softmax'))
model.summary()
#########################################
# 필터 크기(2x2) x 입력 채널(rgb)<전에 있는 것> x 출력 채널<자기가 가진 것> + (bias)=
#########################################
# 2 x 2 x 1 x 10 + 10 = 50
# 3 x 3 x 10 x 3 + 3 = 273
# 3 x 3 x 3 x 5 + 5 = 140
# 2 x 2 x 5 x 3 + 3 = 63
# 2 x 2 x 3 x 2 + 2 = 26
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d (Conv2D)              (None, 8, 7, 10)          50
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 6, 5, 3)           273
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 3, 5)           140
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 3, 2, 3)           63
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 2, 1, 2)           26
# _________________________________________________________________
# flatten (Flatten)            (None, 4)                 0
# _________________________________________________________________
# dense (Dense)                (None, 32)                160
# _________________________________________________________________
# dense_1 (Dense)              (None, 3)                 99
# =================================================================
# Total params: 811
# Trainable params: 811
# Non-trainable params: 0
# _________________________________________________________________