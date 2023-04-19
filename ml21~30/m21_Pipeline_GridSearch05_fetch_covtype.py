import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline, Pipeline # 파이프라인 들어갑니다.
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV


#1. DATA

x, y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=337,
    train_size=0.8
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