# DACNO DDARUNG
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential     
from tensorflow.python.keras.layers import Dense, LSTM     
from sklearn.model_selection import train_test_split, cross_val_predict,cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error # = MSE
import pandas as pd # 전처리(CSV -> 데이터화)
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
                                                #표준정보중심
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. DATA
path = './_data/ddarung/' # path ./은 현재 위치
path_save = './_save/ddarung/'
# Column = Header


# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0) 

# TEST
test_csv = pd.read_csv(path + "test.csv",
                        index_col=0) 

train_csv = train_csv.dropna()


x = train_csv.drop(['count'], axis = 1)
y = train_csv['count']

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.

#2. MODEL
model = RandomForestRegressor()

#3. COMPILE, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x, y, cv=5, n_jobs=-1)
# print(scores)

print('acc: ', scores, '\n cross_val_socre평균: ', round(np.mean(scores), 4)) # mean 평균값
# acc:  [0.77702758 0.75997045 0.81866285 0.74586191 0.76983821] 
#  cross_val_socre평균:  0.7743