import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

#1. DATA
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    shuffle=True,
    random_state=337
)

#2. MODEL
model = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier()
]
model_name = [
    'DTC',
    'RFC',
    'GRC',
    'XGB'
]
for i, v in enumerate(model):
    model = v
    model.fit(x_train,y_train)
    result = model.score(x_test, y_test)
    y_predict = model.predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print(model_name[i], ':', 'ACC: ', acc)
    print(model_name[i], ':', 'importance: ', model.feature_importances_)


# DTC : ACC:  0.9666666666666667
# DTC : importance:  [0.03342386 0.         0.9139125  0.05266364]
# RFC : ACC:  0.9666666666666667
# RFC : importance:  [0.11613174 0.03147137 0.4311152  0.42128168]
# GRC : ACC:  0.9666666666666667
# GRC : importance:  [0.00620961 0.01319906 0.6480249  0.33256644]
# XGB : ACC:  0.9666666666666667
# XGB : importance:  [0.01794496 0.01218657 0.8486943  0.12117416]