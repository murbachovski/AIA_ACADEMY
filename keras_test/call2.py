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

# ISNULL
train_csv = train_csv.dropna()

# x, y SPLIT
x = train_csv.drop([train_csv.columns[-1], train_csv.columns[4]], axis=1)
y = train_csv[train_csv.columns[-1]]
print(x.shape, y.shape) 

# TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=9999,
    stratify=y
)

# SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.unique(y_train, return_counts=True))
print(x_train, x_test)


#2. MODEL
model = Sequential()
model.add(Dense(10, input_dim=x.shape[1]))
model.add(Dense(8, input_dim = 'relu'))
model.add(Dense(6, input_dim = 'relu'))
model.add(Dense(4, input_dim = 'relu'))
model.add(Dense(2, input_dim = 'sigmoid'))

es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=10,
    restore_best_weights=True
) 

#3. COMPILE
model.compile(loss = 'binary_crossentropy', optimizer='adam')
print(x.shape, y.shape)
hist = model.fit(x_train, y_train, epochs=10, batch_size=100, validation_split=0.25, callbacks=[es])

#4. PREDICT
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc:', results[1])
y_predict = model.predict(x_test)
y_predict_acc = np.round(y_predict, axis=1)
y_test_acc = np.round(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict_acc)
f1 = f1_score(y_test_acc, y_predict_acc, average='macro')
print('ACC: ', acc, 'f1: ', f1)

#6. PLT
plt.plot(hist.history['acc'], label='acc', color='red')
plt.plot(hist.history['val_acc'], label='val_acc', color='blue')
plt.plot(hist.history['loss'], label='loss', color='green')
plt.plot(hist.history['val_loss'], label='val_loss', color='yellow')
plt.legend()
plt.show()