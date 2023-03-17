import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense    
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error # ==> MSE

#1. DATA
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike')
#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(train_csv)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(train_csv.shape)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(train_csv.columns)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(train_csv.info())
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(test_csv)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
print(test_csv.shape)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(test_csv.columns)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# print(test_csv.info())
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

#1-4. ISNULL(결측치 처리)
# print(train_csv.isnull().sum())
train_csv.dropna()
# print(train_csv.isnull().sum())

#1-5. DROP(x, y DATA SEPERATE)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
# print(x.shape)
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=32
)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim = x.shape[1]))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 50, batch_size = 1, verbose = 1)

#4.EVALUATE, PREDICT
loss = model.evaluate(x_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2: ', r2)

#5. DEF정의
def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

#6. SUBMISSION_CSV
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path_save + '__submit.csv') # , index= False == 80줄 inddex_col=0 