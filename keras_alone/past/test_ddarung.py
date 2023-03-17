#DACON DDARUNG
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # = MSE
import pandas as pd # 전처리(CSV ===> 데이터화)

#1. DATA
path = './_data/ddarung/' # path ./은 현재 위치
# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0)
print(train_csv)
print(train_csv.shape)
print(train_csv.columns)
print(train_csv.info())
print(train_csv.describe())
print(type[train_csv])
# TEST
test_csv = pd.read_csv(path + "test.csv", index_col=0)
print(test_csv)
print(test_csv.shape)
print(test_csv.columns)
print(test_csv.info())
print(test_csv.describe())
print(type[test_csv])
#결측치 처리########################################################
train_csv = train_csv.dropna()
#결측치 처리########################################################

#train_csv데이터에서 x와 y를 분리########################################################
x = train_csv.drop(['count'], axis=1) # count빼고 가져오기
print(x)
y = train_csv['count'] # count만 가져오기
print(y)
#train_csv데이터에서 x와 y를 분리########################################################

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=32
)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

#2. MODEL
model = Sequential()
model.add(Dense(526, input_dim = 9))
model.add(Dense(124))
model.add(Dense(62))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 10, batch_size = 4, verbose = 1 )

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)
print("RMSE: ", RMSE)

#submission_cdv만들기############################################
print(test_csv.isnull().sum())
print(test_csv.shape)
y_submit = model.predict(test_csv)
print(y_submit)
submission = pd.read_csv(path + "submission.csv", index_col=0)
print("submission: ", submission)
y_submit =  submssion['count']
print(submission)
submission.to_csv(path + "submit.csv")
#submission_cdv만들기############################################
