import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler, OneHotEncoder
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

import datetime #시간을 저장해주는 놈
date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)    # (150, 4) (150,)

#1-2 ONE_HOT_ENCODING(TO_CATEGORICAL)
y = to_categorical(y)

#1-3. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=444,
    stratify=y
)

#1-4. SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) # (120, 4) (30, 4)

#2. Model
model = Sequential()
model.add(Dense(8, input_shape=(4,)))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='softmax')) #iris의 값은 3이다.

# #3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    patience=100,
    mode='auto',
    restore_best_weights=True   
)
mcp = ModelCheckpoint( 
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath= "".join([filepath, 'k28_iris', date, '_', filename]) #restore_best_weights=True가 들어가 있는 모델
)
print(x.shape, y.shape) #(150, 4) (150, 3)
hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_split=0.2, callbacks=[es, mcp])
# # SAVE
# model.save()

# # LOAD
model2 = load_model('_save\MCP\keras27_4\k28_iris0314_1431_0135-0.2100.hdf5')

#4. PREDICT
print("=============================== 1. 기본 출력=======================")
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc: ', results[1])
y_predict = model.predict(x_test)
y_predict_acc = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc: ', acc)
# loss:  0.411452978849411 acc:  0.9333333373069763

#6. PLT
plt.plot(hist.history['val_acc'])
plt.show()

print("=============================== 1. MCP 출력=======================")
results = model2.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc: ', results[1])
y_predict = model2.predict(x_test)
y_predict_acc = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict_acc)
print('acc: ', acc)
# loss:  0.047289784997701645 acc:  0.9666666388511658
