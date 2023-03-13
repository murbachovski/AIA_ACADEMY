# DACNO DDARUNG
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential     
from tensorflow.python.keras.layers import Dense               
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # = MSE
import pandas as pd # 전처리(CSV -> 데이터화)
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler

#1. DATA
path = './_data/kaggle_bike/' # path ./은 현재 위치
path_save = './_save/kaggle_bike/'
# Column = Header

# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0) 
# print(train_csv)
# print(train_csv.shape) # (1459, 10)


# TEST
test_csv = pd.read_csv(path + "test.csv",
                        index_col=0) 
# print(test_csv)
# print(test_csv.shape) # (715, 9)
# print(train_csv.columns) #        'hour_bef_windspeed', 'hour_bef_humidity',    'hour_bef_visibility',
                         #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         #       dtype='object')
# print(train_csv.info())

# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# ---  ------                  --------------  -----

# print(train_csv.describe()) # [8 rows x 10 columns]

# print("TYPE",type[train_csv])



############################결측치 처리#############################
#1. 결측치 처리 - 제거
# print(train_csv.isnull().sum())  #중요하답니다.
train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())
# print(train_csv.info())
# print(train_csv.shape) # (1328, 10)


#########################################train_csv데이터에서 x와 y를 분리했다.
#########이게 중요합니다#######
x = train_csv.drop(['count', 'casual', 'registered'], axis = 1)
# print(x)
y = train_csv['count']
# print(y)
#########################################train_csv데이터에서 x와 y를 분리했다.

# MinMaxScaler()                # r2: 0.2804708249711828
scaler = MinMaxScaler() 
scaler.fit(x)
x = scaler.transform(x)
# StandardScaler()              # r2: 0.5803998528889721
# scaler = StandardScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# MaxAbsScaler()                # r2: 0.5758372715749012
# scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)
# RobustScaler()                # r2: 0.5702060698020737
# scaler = MaxAbsScaler()
# scaler.fit(x)
# x = scaler.transform(x)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    train_size=0.7,
    random_state=777
)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# print(x_train.shape, x_test.shape) # (1021, 9) (438, 9)   ---> (929, 9 ) (399, 9)
# print(y_train.shape, y_test.shape) # (1021, ) (438, )     ---> (929, ) (399, )
 
#2. MODEL
model = Sequential()
model.add(Dense(526, input_dim = x.shape[1]))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
es = EarlyStopping(
    monitor='val_loss',
    patience=20,
    mode='min',
    verbose=1,
    restore_best_weights=True
    )
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 50, verbose=1, validation_split=0.2,
          callbacks=[es]
          )

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2:", r2)


def RMSE(y_test, y_predict):                #RMSE 함수 정의
    return np.sqrt(mean_squared_error(y_test, y_predict)) # np.sqrt = route
rmse = RMSE(y_test, y_predict)             #RMSE 함수 사용
print("rmse: ", rmse)

######### Let's make a submission_csv ##########
# print(test_csv.isnull().sum()) # 요기도 결측치가 있네~?
# test_csv = test_csv.dropna() # 결측치를 제거해주면 index가 맞지 않아서 error가 나옵니다.
# print(test_csv.shape)

y_submit = model.predict(test_csv)
# # print(y_submit)
submission = pd.read_csv(path + "submission.csv",index_col=0)
# print("submission: ", submission)
submission["count"] = y_submit
# print(submission)
submission.to_csv(path_save + "submit_new.csv")
#그림 그리기
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9, 6)) 
plt.plot(hist.history['loss'], marker = '.', c='red', label ='loss')
plt.plot(hist.history['val_loss'], marker = '.', c='blue', label ='val_loss')
plt.title('ddarung')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')
plt.legend()
plt.grid()
plt.show()

