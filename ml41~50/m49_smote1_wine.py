import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# 1. DATA
datasets = load_wine()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape)
# (178, 13) (178,)
# print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))

# print(pd.Series(y).value_counts().sort_index()) # numpy to Series # sort_index해주면 인덱스로 배열해줍니다.
# 1    71
# 0    59
# 2    48
# dtype: int64

# 0    59
# 1    71
# 2    48
# dtype: int64
# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
# print(x.shape) # (178, 13)
# print(y.shape) # (178,)
x = x[:-25]
y = y[:-25]
# print(x.shape) # (153, 13)
# print(y.shape) # (153,)
# print(y)
# print(pd.Series(y).value_counts().sort_index())
# 0    59
# 1    71
# 2    23
# dtype: int64

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle=True, random_state=3377, stratify=y
)
# print(pd.Series(y_train).value_counts().sort_index())
# 0    44
# 1    53
# 2    17
# dtype: int64

# 2. MODEL
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=3377)

# 3. COMPILE
model.fit(x_train, y_train)

#4. PREDICT
score = model.score(x_test, y_test)
print("MODEL_SCORE: ", score)
y_pred = model.predict(x_test)
print('ACC: ', accuracy_score(y_test, y_pred))
print("F1_SCORE_MACRO: ", f1_score(y_test, y_pred, average='macro'))      # 가장 많이 쓰는 것 # 이진분류에서 사용합니다.
# print("F1_SCORE: ", f1_score(y_test, y_pred, average='micro'))      # average로 통해 다중분류에서도 사용할 수 있도록 했다.
# print("F1_SCORE: ", f1_score(y_test, y_pred, average='weighted'))   # 

print('====================================SMOTE 적용 후!!!!!!!!!!!!!!!!!!!!') # 통상적으로 향상되는 경우가 많다.
smote = SMOTE(random_state=3377, k_neighbors=8)
# print(x_train.shape) # (114, 13)
# print(y_train.shape) # (114,)
x_train, y_train = smote.fit_resample(x_train, y_train) # 최대값의 맞춰서 증폭됩니다. # y_test에 증폭하지 않은 이유는 조작 관련
# print(x_train.shape) # (159, 13)
# print(y_train.shape) # (159,)
# print(pd.Series(y_train).value_counts().sort_index())
# 0    53
# 1    53
# 2    53
# dtype: int64

# 2-2. MODEL
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=3377) 

# 3-2. COMPILE
model.fit(x_train, y_train)

# 4-2. PREDICT
score = model.score(x_test, y_test)
print("MODEL_SCORE: ", score)
y_pred = model.predict(x_test)
print('ACC: ', accuracy_score(y_test, y_pred))
print("F1_SCORE_MACRO: ", f1_score(y_test, y_pred, average='macro'))      # 가장 많이 쓰는 것 # 이진분류에서 사용합니다.
# print("F1_SCORE: ", f1_score(y_test, y_pred, average='micro'))      # average로 통해 다중분류에서도 사용할 수 있도록 했다.
# print("F1_SCORE: ", f1_score(y_test, y_pred, average='weighted'))   # 
