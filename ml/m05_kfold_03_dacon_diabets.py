import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, MaxAbsScaler, StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier # 분류

import datetime #시간을 저장해주는 놈
date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = ('./_data/dacon_diabetes/')
path_save = ('./_save/dacon_diabetes/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape, test_csv.shape)    # (652, 9) (116, 8)

#1-2 x, y SPLIT
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.

# #2. Model
model = RandomForestClassifier()

#3. COMPILE, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x, y, cv=5, n_jobs=-1) # n_jobs = CPU 최대치
# print(scores)

print('r2: ', scores, '\n cross_val_socre평균: ', round(np.mean(scores), 4)) # mean 평균값