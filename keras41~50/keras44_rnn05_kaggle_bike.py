import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error # MSE
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib
import datetime #시간을 저장해주는 놈
from sklearn.preprocessing import MinMaxScaler

date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/kaggle_bike/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL(결측치 처리)
train_csv = train_csv.dropna()

#1-5. (x, y DATA SPLIT)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)   #?
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=32
)

print(x_train.shape, x_test.shape) #(7620, 8) (3266, 8)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(7620, 8, 1)
x_test = x_test.reshape(3266, 8, 1)

#2. MODEL
model = Sequential()
model.add(LSTM(256, input_shape=(8, 1)))
model.add(Dropout(0.6))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

#3. COMPILE
model.compile(loss ='mse', optimizer='adam')
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    restore_best_weights=True,
    patience=10,
    verbose=1
    )
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    filepath= "".join([filepath + 'kaggle_bike', date, '_-_', filename]),
    verbose=1,
)
hist = model.fit(x_train, y_train, epochs = 1000, batch_size=200, verbose=1, validation_split=0.2, callbacks=[es])

# model2 = load_model('_save\MCP\keras27_4\kaggle_bike.3047.hdf5')

print("=============================== 1. 기본 출력=======================")
#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2: ', r2)
# loss:  28339.6953125
# r2:  0.15957136764888924

print("=============================== 2. MCP 출력=======================")
#4. EVALUATE, PREDICT
# loss = model2.evaluate(x_test, y_test)
# print('loss: ', loss)
# y_predict = model2.predict(x_test)
# r2 = r2_score(y_test, y_predict)
# print('r2: ', r2)
# loss:  28339.6953125
# r2:  0.15957136764888924
# rmse:  10.815610773876188

#5 DEF
# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_absolute_error(y_test, y_predict))
# rmse = RMSE(y_test, y_predict)
# print("rmse: ", rmse)

#7. PLT
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], marker = '.', c ='red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.',
         c='blue', label = 'val_loss'
)
plt.title('KAGGLE_BIKE')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()
# loss:  34283.78125