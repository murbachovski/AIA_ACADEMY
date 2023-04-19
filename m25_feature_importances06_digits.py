# 총 10개 데이터 셋
# 10개의 파일을 만든다.
# 피처를 한개씩 삭제하고 성능비교
# 모델은 RF로만 한다.

# 총 10개 데이터 셋
# 10개의 파일을 만든다.
# 피처를 한개씩 삭제하고 성능비교
# 모델은 RF로만 한다.

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier


from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
#1. DATA

x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=337,
    train_size=0.8
)

model_list = [
    RandomForestRegressor(),
]

for i in model_list:
    model = i
    #3. COMPILE
    model.fit(x_train, y_train)

    #4. PREDICT
    result = model.score(x_test, y_test)
    print('model: ', model, 'result: ',  result, 'model_feature: ', model.feature_importances_)
    print('==================================')

# model:  RandomForestRegressor() result:  0.8473669847935589 model_feature:  [0.00000000e+00 1.14962314e-03 5.39088671e-03 5.53429110e-03
#  5.21616197e-03 6.82140364e-03 2.26265200e-03 3.61855486e-04
#  1.11922042e-04 2.64430523e-03 1.18402260e-02 4.91695760e-03
#  2.75996640e-02 1.32925149e-02 3.93878838e-03 1.46161979e-04
#  1.76242302e-04 1.87967222e-03 2.11530936e-02 1.24020723e-02
#  8.58747954e-02 6.74513405e-02 8.22242872e-03 7.51273520e-05
#  1.38868604e-04 3.67849576e-03 2.44412425e-02 4.58588705e-02
#  6.13011463e-02 5.60553824e-02 1.48841037e-02 0.00000000e+00
#  0.00000000e+00 3.14030754e-02 2.04828140e-02 4.00756697e-02
#  9.39338084e-02 5.73880766e-03 7.20666604e-03 0.00000000e+00
#  8.07796320e-07 4.20148804e-03 2.43754981e-02 1.52381336e-02
#  6.51986033e-03 6.50680284e-03 8.69897819e-03 7.31689094e-05
#  0.00000000e+00 1.69679968e-03 5.95268955e-03 1.67695405e-02
#  1.05868916e-01 1.07640714e-02 7.04532615e-03 4.80204195e-04
#  0.00000000e+00 9.46028002e-04 6.16535246e-03 6.27962629e-03
#  1.76490618e-02 1.47443224e-02 3.36802936e-02 1.26818927e-02]
# ==================================