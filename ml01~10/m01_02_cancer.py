import numpy as np
from sklearn.datasets import load_breast_cancer

#1. DATA
# datasets = load_iris()
# x = datasets.data
# y = datasets.target() # == y = datasets['target']

x, y = load_breast_cancer(return_X_y=True)
print(x.shape, y.shape) # (150, 4) (150,)
print()

#2. MODEL
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from sklearn.svm import LinearSVC # 머신러닝 들어갑니다.
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor

# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(4, )))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3, activation='softmax'))
# model = LinearSVC(C=500) # 알고리즘 연산이 포함되어 있다. C가 클수록 곡선을 그려준다. 작을수록 직선
# model = LogisticRegression() # 이진 분류
# model = DecisionTreeRegressor() #회귀 모델
model = DecisionTreeClassifier()
# model = RandomForestRegressor()

#3. COMPILE
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc']) # 원핫 안 해줬을 경우 가능하다. 0부터 시작하는지 확인하기
# model.fit(x,y, epochs=100, validation_split=0.2)
model.fit(x,y)

#4. PREDICT
# results = model.evaluate(x, y)
results = model.score(x,y)
print(results)
# 0.9797003739231541