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
#1. DATA

x, y = load_iris(return_X_y=True)

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
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier()
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


# DecisionTreeClassifier()  [0.01671193 0.03342386 0.9139125  0.03595171]
# RandomForestClassifier()  [0.11470372 0.02867922 0.41892914 0.43768792]
# GradientBoostingClassifier()  [0.00199761 0.01743598 0.69241162 0.28815479]
# XGBClassifier()   [0.01794496 0.01218657 0.8486943  0.12117416]