# 선형 회귀 / 이진 / 다중
# relu, linear  / sigmoid / softmax
# MSE, MAE / binary_crossentropy / categorical_crossentropy
# ONE_HOT_ENCODING은 분류에서만 사용합니다.
# stratify=y 분류에서만 사용합니다.
# scaler는 성능 향상에 도움이 됨으로 모든 곳에 사용이 가능하다.
# acc = 분류에서 사용합니다.
# metrics=['acc']는 지표를 확인하기 위한 것, accuracy_score와는 다른 것이다.
# np.round() = 이진분류
# np.argmax() = 다중분류


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. DATA
dataset = fetch_california_housing()
x = dataset.data
y = dataset.target
print(x.shape, y.shape) #(20640, 8) (20640,)

#2. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=222, 
)
print(x_train.shape, y_train.shape) #(16512, 8) (16512,)
print(x_test.shape, y_test.shape)   #(4128, 8) (4128,)

#. SCALER 이곳에서 한번만 해줍니다.
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape)

# 3. MODEL
model = Sequential()
model.add(Dense(12, input_dim=x.shape[1])) #input_dim=13
model.add(Dense(6))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1))

## 동일한 모델 구성

# model = Model()구성
# input1 = Input(shape=(8,))
# dense1 = Dense(12)(input1)
# dense2 = Dense(6)(dense1)
# dense3 = Dense(4, activation='relu')(dense2)
# dense4 = Dense(2, activation='relu')(dense3)
# output1 = Dense(1)(dense4)
# model = Model(inputs=input1, outputs=output1)

# 4. COMPILE
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(
    monitor='val_loss',
    patience=10,
    mode='min',
    restore_best_weights=True,
    verbose=1
)
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/MCP/keras27_ModelCheckPoint1.hdf5'
)
hist = model.fit(x_train, y_train, epochs=10000, validation_split=0.2, batch_size=50, callbacks=[es, mcp])

# # model.save()
# model.save('./_save/23_03_13.h5')

# model = load_model()
# model = load_model('./_save/23_03_13.h5')

#5. PREDICT
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss:', results[0], 'acc:', results[1], 'r2:', r2)


## 이 가중치를 저장.
# loss: 0.4414449632167816 acc: 0.0033914728555828333 r2: 0.659702422302971
## load_model()로 불러온 결과값
# loss: 0.4414449632167816 acc: 0.0033914728555828333 r2: 0.659702422302971

import matplotlib.pyplot as plt
plt.plot(hist.history['val_loss'])
plt.show()

