import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler는 이상치도 잡아준다
from xgboost import XGBClassifier, XGBRegressor


x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=337,
    shuffle=True,
    test_size=0.2,
    stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = {
            'n_estimators' : 100, # epochs
            'learning_rate' : 5, # 
            'max_depth' : 2, #
            'gamma' : 0, #
            'min_child_weight' : 1, #
            'subsample' : 1, # 
            'colsample_bytree' : 1,
            'colsample_bylevel' : 1,
            'colsample_bynode' : 1,
            'reg_alpha' : 0,
            'reg_lambda' : 1,
            'random_state' : 337
}

# 2. MODEL
model = XGBClassifier(**parameters)

# 3. COMPILE
model.fit(x_train, y_train,
          eval_set=[(x_train, y_train), (x_test, y_test)],
          early_stopping_rounds=10,
          verbose=True,
          )

results = model.score(x_test, y_test)
print('FINISH SCORE: ', results)

# *args를 사용하면 **kwargs모든 인수를 미리 정의하지 않고도
# 다양한 수의 인수를 처리할 수 있는 함수를 작성할 수 있으므로 코드를 더 유연하고 재사용 가능하게 만들 수 있습니다.

# * 변수 다양하게 받으려고
# ** dic형태 key, value로 받으려고 