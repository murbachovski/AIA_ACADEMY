import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor


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

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

model_list = [
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
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
    # y_predict = model.predict(x_test)
    # acc = accuracy_score(y_test, y_predict)
    print('model: ', model, 'result: ',  result, 'model_feature: ', model.feature_importances_)
    print('==================================')
    # print(model, '', model.feature_importances_) # TREE계열에만 존재합니다. feature...

# model_feature:  [0.00000000e+00 7.00646294e-05 2.96583333e-03 2.56435415e-03
