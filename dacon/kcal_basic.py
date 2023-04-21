import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import joblib
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model, Model
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout,Input, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping as es
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder # SCALER
import matplotlib.pyplot as plt 
import datetime #시간을 저장해주는 놈
from tqdm import tqdm_notebook
import time
# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# DATA
train_csv = pd.read_csv(path+'train.csv', index_col=0)
test_csv = pd.read_csv(path+'test.csv', index_col=0)
submission = pd.read_csv(path+'sample_submission.csv')
# print(train_data.shape, test_data.shape) # (7500, 10) (7500, 9)
# print(train_csv.info())
 #   Column                    Non-Null Count  Dtype
# ---  ------                    --------------  -----
#  0   Exercise_Duration         7500 non-null   float64
#  1   Body_Temperature(F)       7500 non-null   float64
#  2   BPM                       7500 non-null   float64
#  3   Height(Feet)              7500 non-null   float64
#  4   Height(Remainder_Inches)  7500 non-null   float64
#  5   Weight(lb)                7500 non-null   float64
#  6   Weight_Status             7500 non-null   object
#  7   Gender                    7500 non-null   object
#  8   Age                       7500 non-null   int64
#  9   Calories_Burned           7500 non-null   float64

cols = ['Weight_Status', 'Gender']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_csv[col]=le.fit_transform(train_csv[col])
    test_csv[col]=le.fit_transform(test_csv[col])

# ISNULL
train_csv = train_csv.dropna()

# x, y SPLIT
x = train_csv.drop(['Calories_Burned'], axis=1)
y = train_csv['Calories_Burned']
# print(x.shape, y.shape) # (7500, 9) (7500,)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=222,
    test_size=0.2
)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)
# (6000, 9) (6000,)
# (1500, 9) (1500,)

x_train = x_train.reshape(6000, 3, 3)
x_test = x_test.reshape(1500, 3, 3)
# print(x_train.shape, y_train.shape) # (6000, 3, 3) (6000,)
# print(x_test.shape, y_test.shape)   # (1500, 3, 3) (1500,)

#2. 모델구성
# model = Sequential()
# model.add(Dense(256, input_shape=(9,)))
# model.add(Dropout(0.4))
# model.add(Dense(128, activation=LeakyReLU(0.15)))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation=LeakyReLU(0.15)))
# model.add(Dropout(0.1))
# model.add(Dense(64, activation=LeakyReLU(0.15)))
# model.add(Dense(1))

# model = Sequential()
# model.add(LSTM(128, input_shape=(3, 3), return_sequences=True))
# model.add(LSTM(64, return_sequences=True, activation='relu'))
# model.add(GRU(32, return_sequences=True, activation='relu'))
# model.add(GRU(16, return_sequences=True, activation='relu'))
# model.add(Dense(1))

model = Sequential()
model.add(LSTM(256, input_shape=(3, 3), return_sequences=True))
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

# 3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(
    monitor='val_loss',
    patience=100,
    mode='min',
    restore_best_weights=True,
    verbose=1
)

mcp = ModelCheckpoint(
    monitor='val_loss',
    mode='auto',
    verbose=1,
    save_best_only=True,
    filepath='./_save/dacon_kcal/_MCP.hdf5'
)

rdp = ReduceLROnPlateau(
    monitor='val_acc',
    mode='auto',
    verbose=1,
    patience=3 
)

hist = model.fit(x_train, y_train, epochs=1, validation_split=0.2, batch_size=200, callbacks=[es, mcp, rdp])

x_train = x_train.reshape(6000, 3*3)
x_test = x_test.reshape(1500, 3*3)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

#5. DEF정의
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

#6. SUBMISSION_CSV
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
submission['Calories_Burned'] = y_submit
submission.to_csv(save_path + 'dacon_kaggle_submit.csv')

import matplotlib.pyplot as plt
import seaborn as sns
print(test_csv.corr())
plt.figure(figsize=(10,8))
sns.set(font_scale=1.2)
sns.heatmap(train_csv.corr(), square=True, annot=True, cbar=True)
plt.show()

print('loss: ', loss, 'r2: ', r2, "rmse: ", rmse)
