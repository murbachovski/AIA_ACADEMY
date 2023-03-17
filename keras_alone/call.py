import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Input, load_model, Model, save_model
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, MaxAbsScaler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import datetime #시간으로 저장해주는 고마운 녀석
date = datetime.datetime.now()
# print(date) #2023-03-14 22:02:28.099457
date = date.strftime('%m%d_%M%M')
filepath = ('./_save/call/')
filename = '{epoch:04d}_{val_loss:.4f}_{val_acc:.4f}.hdf5'
#test
#1. DATA
path = ('./_data/call/')
path_save = ('./_save/call/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape, test_csv.shape) #(30200, 14) (12943, 13)
print(train_csv, test_csv)

# ISNULL
train_csv = train_csv.dropna()

# x, y SPLIT
x = train_csv.drop([train_csv.columns[-1]], axis=1)
y = train_csv[train_csv.columns[-1]]
print(x.shape, y.shape) #(30200, 12) (30200,)

# ONE_HOT
y = to_categorical(y)

# TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=2222,
    stratify=y
)

# SCALER
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.unique(y_train, return_counts=True))

#2. MODEL
# model = Sequential()
# model.add(Dense(256, input_shape=(12,)))
# model.add(Dropout(0.3))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(2, activation='softmax'))

model = load_model('_save\call\dacon_call0316_4343_0001_0.0875_0.9663.hdf5')

#3. COMPILE
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_acc',
    mode='auto',
    patience=50,
    restore_best_weights=True
)
mcp = ModelCheckpoint(
    monitor='val_acc',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath="".join([filepath, 'dacon_call', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.3, batch_size=2000, callbacks=[es])
# model.save('./_save/call/call_model_of_best.h5')
# model = load_model('_save\call\call_model_best.h5')
# es = EarlyStopping(
#     monitor='val_acc',
#     mode='auto',
#     patience=50,
#     restore_best_weights=True
# ) 
# mcp = ModelCheckpoint(
#     monitor='val_acc',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     filepath="".join([filepath, 'dacon_call', date, '_', filename])
# )
# hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.3, batch_size=9000, callbacks=[es, mcp])
# model.save('./_save/call/call_model_best.h5')

#4. PREDICT
results = model.evaluate(x_test, y_test)
print('loss: ', results[0], 'acc:', results[1])
y_predict = model.predict(x_test)
#print(y_predict)
#print(y_predict.shape)
y_predict_acc = np.argmax(y_predict, axis=1)
y_test_acc = np.argmax(y_test, axis=1)
acc = accuracy_score(y_test_acc, y_predict_acc)
f1 = f1_score(y_test_acc, y_predict_acc, average='macro')
print('ACC: ', acc, 'f1: ', f1)

#5. SUBMIT
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