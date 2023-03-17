import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Input, load_model, Model, save_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import datetime #시간으로 저장해주는 고마운 녀석
date = datetime.datetime.now()

date = date.strftime('%m%d_%M%M')
filepath = ('./_save/call/')
filename = '{epoch:04d}_{val_loss:.4f}_{val_acc:.4f}.hdf5'

#1. DATA
path = ('./_data/call/')
path_save = ('./_save/call/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape, test_csv.shape) 
print(train_csv, test_csv)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_shape=(11,)))

#5. SUBMIT
test_csv = test_csv.drop(test_csv[4], axis=1)
test_csv_sc = scaler.transform(test_csv)
y_submit = model.predict(test_csv_sc)
y_submit = np.argmax(y_submit, axis=1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission[train_csv.columns[-1]] = y_submit
submission.to_csv(path_save + 'call_submit4.csv')

#6. PLT
plt.plot(hist.history['acc'], label='acc', color='red')
plt.plot(hist.history['val_acc'], label='val_acc', color='blue')
plt.plot(hist.history['loss'], label='loss', color='green')
plt.plot(hist.history['val_loss'], label='val_loss', color='yellow')
plt.legend()
plt.show()