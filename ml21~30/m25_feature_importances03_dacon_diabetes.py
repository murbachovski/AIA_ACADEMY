# 총 10개 데이터 셋
# 10개의 파일을 만든다.
# 피처를 한개씩 삭제하고 성능비교
# 모델은 RF로만 한다.

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd

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

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=337,
    train_size=0.8
)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

model_list = [
    RandomForestClassifier(),
]

#2. MODEL # TREE계열
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

# model = make_pipeline(RobustScaler(), LocalOutlierFactor())
for i in model_list:
    model = i
    #3. COMPILE
    model.fit(x_train, y_train)

    #4. PREDICT
    result = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print('model: ', model, 'result: ',  result, 'ACC: ', acc, 'model_feature: ', model.feature_importances_)
    print('==================================')
    # print(model, '', model.feature_importances_) # TREE계열에만 존재합니다. feature...


# model:  RandomForestClassifier() result:  0.7633587786259542 ACC:  0.7633587786259542 model_feature:  [0.09277669 0.24514886 0.08887135 0.07113605 0.07062359 0.16541349
#  0.12462749 0.14140248]