# 총 10개 데이터 셋
# 10개의 파일을 만든다.
# 피처를 한개씩 삭제하고 성능비교
# 모델은 RF로만 한다.

# 총 10개 데이터 셋
# 10개의 파일을 만든다.
# 피처를 한개씩 삭제하고 성능비교
# 모델은 RF로만 한다.

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
#1. DATA

x, y = load_wine(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=337,
    train_size=0.8
)

model_list = [
    RandomForestClassifier(),
]

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

# model:  RandomForestClassifier() result:  0.9444444444444444 ACC:  0.9444444444444444 model_feature:  [0.11095448 0.03042776 0.023477   0.0242381  0.02818454 0.05092617
#  0.16227875 0.01704189 0.01555511 0.14912622 0.07748464 0.0965341
#  0.21377123]
# ==================================