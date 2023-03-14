# DACNO DDARUNG
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model     
from tensorflow.python.keras.layers import Dense, Dropout          
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # = MSE
import pandas as pd # 전처리(CSV -> 데이터화)
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import datetime #시간을 저장해주는 놈

date = datetime.datetime.now()
print(date) # 2023-03-14 11:10:57.138016
date = date.strftime('%m%d_%H%M')
print(date) # 0314_1115
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = './_data/ddarung/' # path ./은 현재 위치
path_save = './_save/ddarung/'

# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0) 


# TEST
test_csv = pd.read_csv(path + "test.csv",
                        index_col=0) 

############################결측치 처리#############################
#1. 결측치 처리 - 제거
train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

scaler = MaxAbsScaler()
scaler.fit(x)
x = scaler.transform(x)
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
model.add(Dense(526, input_dim = 9))
model.add(Dropout(0.3))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
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
mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    save_best_only=True,
    verbose=1,
    filepath= ''.join([filepath + 'k28_DDARUNG', date, '_', filename])
)
hist = model.fit(x_train, y_train, epochs = 100, batch_size = 50, verbose=1, validation_split=0.2,
          callbacks=[es]
)

model2 = load_model('_save\MCP\keras27_4\k28_DDARUNG0314_1532_0062-2431.6833.hdf5')
print("=============================== 1. 기본 출력=======================")
#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2:", r2)
# loss: 2821.78662109375
# r2: 0.5687333301384413


print("=============================== 1. MCP 출력=======================")
loss = model2.evaluate(x_test, y_test)
print("loss:", loss)
y_predict = model2.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2:", r2)

def RMSE(y_test, y_predict):           
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)          
print("rmse: ", rmse)
# loss: 2821.78662109375
# r2: 0.5687333301384413
# rmse:  53.12049159865407


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

