import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier # 분류


#1. DATA
x, y = load_digits(return_X_y=True)

# x_train, x_test, y_train, y_test = train_test_split(
#     x,
#     y,
#     random_state=222,
#     shuffle=False,
#     test_size=0.2
# )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=22) # cross_val_score내용들을 정리한 것.

#2. MODEL
model = RandomForestClassifier()

#3. COMPILE, 훈련, 평가, 예측
# scores = cross_val_score(model, x, y, cv=kfold)
scores = cross_val_score(model, x, y, cv=5)
# print(scores)

print('r2: ', scores, '\n cross_val_socre평균: ', round(np.mean(scores), 4)) # mean 평균값
# r2:  [0.93055556 0.89166667 0.95543175 0.95821727 0.92200557] 
#  cross_val_socre평균:  0.9316