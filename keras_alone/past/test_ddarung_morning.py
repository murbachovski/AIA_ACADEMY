import numpy as np    
import pandas as pd     
import tensorflow as tf      
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error # => MSE
from tensorflow.keras.models import Sequential   
from tensorflow.keras.layers import Dense   

#1. DATA
path = ('./_data/ddarung/') # 데이터 경로 설정

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0) # train.csv 불러와서 변수 처리

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0) # test.csv 불러와서 변수 처리

#1-4. ISNULL(결측치 제거)
train_csv = train_csv.dropna() # dropna()를 이용해서 결측치 제거

#1-5. DROP(x, y데이터 분리)
x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=32,
    train_size=0.7
)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim = 9))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 32, verbose = 1)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print('loss: ', loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

#5. DEFR(MSE함수 정의)
def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

#6. SUBMISSION_CSV
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv', index_col=0)
submission['count'] = y_submit
submission.to_csv(path + 'submit.csv') # 저장경로