import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline # 파이프라인 들어갑니다.
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
import pandas as pd

#1. DATA


# 1. 데이터
filepath = ('./_save/MCP/keras27_4/')
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

#1. DATA
path = ('./_data/dacon_diabetes/')
path_save = ('./_save/dacon_diabetes/')

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(train_csv.shape, test_csv.shape)    # (652, 9) (116, 8)

#1-2 x, y SPLIT
x = train_csv.drop(['Outcome'], axis=1)
y = train_csv['Outcome']   

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=337, test_size=0.2, #stratify=y # stratify=y 사용함으로써 각 라벨값이 골고루 분배된다 
)   

parameters = [
    {'rf__n_estimators': [100, 200], 'rf__max_depth' : [6, 10, 12]}
]

#2. MODEL
pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestClassifier())]) #TypeError: 'StandardScaler' object is not iterable 해결방법 튜플로 만들어주기!!

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)

#3. COMPILE
model.fit(x_train, y_train)

#4. PREDICT
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('model: ', result, 'ACC: ', acc)
# model:  0.7709923664122137 ACC:  0.7709923664122137