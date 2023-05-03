#m51_bagging1 카피

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.ensemble import VotingClassifier #투표
from sklearn.preprocessing import PolynomialFeatures
#1. 데이터
x,y  = load_wine(return_X_y=True)

POLY = PolynomialFeatures(degree=2)
x_POLY = POLY.fit_transform(x)
# print(x_POLY.shape) # (150, 15)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle= True, train_size=0.8, random_state=1253
)

# scaler = StandardScaler()
# x_train =  scaler.fit_transform(x_train)
# x_test =  scaler.fit_transform(x_test)

#2. 모델
model = RandomForestClassifier()

#3. 훈련
model.fit(x_train,y_train)

#4. 평가, 예측
y_pred = model.predict(x_test)
print('model.score : ', model.score(x_test,y_test))
print("acc : ", accuracy_score(y_test,y_pred))