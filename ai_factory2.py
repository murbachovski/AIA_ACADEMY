import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler, OneHotEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

import datetime #시간을 저장해주는 놈
date = datetime.datetime.now()
# print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
# print(date) # 0314_1115
filepath = ('./_save/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = ('./_data/ai_factory/')
path_save = ('./_save/ai_factory/')

train_csv = pd.read_csv(path + 'train_data.csv', index_col=0)
test_csv = pd.read_csv(path + 'test_data.csv', index_col=0)
# print(train_csv.shape, test_csv.shape) # (2463, 7) (7389, 7)

#1-1. ISNULL
train_csv = train_csv.dropna()
# print(train_csv.shape) # (652, 9)

#1-2 x, y SPLIT
x = train_csv.drop(['type'], axis=1)
y = train_csv['type']
print(x, y)
print(x.shape, y.shape) # (2463, 6) (2463,)

#2. MODEL
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=(2463, 7)))
model.add(Dense(1, activation='sigmoid'))

#2. COMPILE
model.compile(loss='binary_crossentropy', optimizer='adam')
hist = model.fit(x, y, epochs=10)

#4. PREDICT
results = model.evaluate(x, y)
# print('loss: ', results[0], 'acc: ', results[1])

#5. SUBMISSION_CSV
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'answer_sample.csv', index_col=0)
submission['type'] = y_submit
submission.to_csv(path_save + 'aifactory'+ date + '.csv')