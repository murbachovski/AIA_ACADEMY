import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline # 파이프라인 들어갑니다.
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
import pandas as pd


#1. DATE
path = ('./_data/kaggle_bike/')
path_save = ('./_save/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0)

#1-3. TEST
test_csv =  pd.read_csv(path + 'test.csv', index_col=0)

#1-4. ISNULL(결측치 처리)
train_csv = train_csv.dropna()

#1-5. (x, y DATA SPLIT)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT()
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=32
)

scaler = MinMaxScaler() 
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

parameters = [
    {'rf__n_estimators': [100, 200], 'rf__max_depth' : [6, 10, 12]}
]

#2. MODEL
pipe = Pipeline([('std', StandardScaler()), ('rf', RandomForestRegressor())]) #TypeError: 'StandardScaler' object is not iterable 해결방법 튜플로 만들어주기!!

model = GridSearchCV(pipe, parameters, cv=5, verbose=1, n_jobs=-1)

#3. COMPILE
model.fit(x_train, y_train)

#4. PREDICT
result = model.score(x_test, y_test)
# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
print('model: ', result)
# model:  0.3514061149757388