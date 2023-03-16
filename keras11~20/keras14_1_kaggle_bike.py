import pandas as pd
import numpy as np 
import tensorflow as tf  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

#1. PATH
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0) #index_col이 필요한 상황일까?
print(train_csv.shape)

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv.shape)
print(train_csv.columns)
print(test_csv.columns)

#1-4. ISNULL(결측치 처리)
print(train_csv.isnull().sum()) # 결측치가 다 없네?(꽉 차 있는 상황인가?)
train_csv = train_csv.dropna()

#1-5. DROP(x, y데이터 분리)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1) # 이곳에서 데이터를 같이 제거해준다.
y = train_csv['count']
print('x.columns: ', x.columns)
print(train_csv.columns)


#1-6. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=32,
    train_size=0.8
)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim = x.shape[1])) # x.shape[1]
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32, activation = 'linear')) # 
model.add(Dense(16, activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1))                         # activation = 'relu'  음수를 0으로 만들어준다.

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 10, verbose = 1)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
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
submission.to_csv(path_save + '_submit.csv')
