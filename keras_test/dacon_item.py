import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential,load_model, Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten, Input
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
import datetime

date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#DATA
path = ('./_data/dacon_item/')
path_save = ('./_save/dacon_item/')

train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')

# print(train_csv.shape, test_csv.shape) # (31684, 5) (7920, 4)

# 결측치 없음.

# LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
cols = ['송하인_격자공간고유번호', '수하인_격자공간고유번호', '물품_카테고리']
for col in tqdm(cols):
    le = LabelEncoder()
    train_csv[col] =le.fit_transform(train_csv[col])
    test_csv[col] =le.fit_transform(test_csv[col])

# x, y SPLIT
x = train_csv.drop(['운송장_건수'], axis=1)
y = train_csv['운송장_건수']

# TRAIN_TEST_SPLIT
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=2222,
    test_size=0.025
)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape, x_test.shape) # (30891, 4) (793, 4)

#MODEL
model = Sequential()
model.add(Dense(256, input_shape=(4,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1))

#COMPILE
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    patience=100,
    mode='auto',
    restore_best_weights=True
)
hist = model.fit(x_train, y_train, epochs=1000, validation_split=0.025, batch_size=800, callbacks=[es])

model.save('./_save/dacon_item_model.h5')
# model = load_model('')

#PREDICT
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('loss:', results[0], 'r2:', r2)

def RMSE(y_test, y_predict):                #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt = route
rmse = RMSE(y_test, y_predict)             #RMSE 함수 사용
print("rmse: ", rmse)

#SUBMISSION_CSV
test_csv_sc = scaler.transform(test_csv)
y_submit = model.predict(test_csv_sc)
y_submit = np.argmax(y_submit, axis=1)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['운송장_건수'] = y_submit
submission.to_csv(path_save + 'dacon_item'+ date + '.csv')

#6. PLT
plt.plot(hist.history['val_loss'],label='val_loss', color='red')
plt.legend()
plt.show()