# 회귀로 맹그러
# 회귀데이터 올인!!! 포문
# scaler 6개 올인!!! 포문

from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

# 1. DATA
x, y = fetch_california_housing(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    train_size=0.8,
    random_state=337,
    # stratify=y
)
sc_list = [
    StandardScaler(),
    RobustScaler(),
    MinMaxScaler(),
    MaxAbsScaler(),
    QuantileTransformer(),
    PowerTransformer()
]
for scaler in sc_list:
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 2. MODEL
    model = RandomForestRegressor()

    # 3. COMPILE
    model.fit(x_train, y_train)

    # 4. PREDICT
    print('RESULTS: ', round(model.score(x_test, y_test)))
# RESULTS:  1
# RESULTS:  1
# RESULTS:  1
# RESULTS:  1
# RESULTS:  1
# RESULTS:  1