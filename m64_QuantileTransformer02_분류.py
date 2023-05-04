# 회귀로 맹그러
# 회귀데이터 올인!!! 포문
# scaler 6개 올인!!! 포문

from sklearn.datasets import load_iris, load_breast_cancer, load_digits, load_diabetes, load_wine
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
import numpy as np
import pandas as pd
# from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split

# 1. DATA
data_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_digits(return_X_y=True),
    load_diabetes(return_X_y=True),
    load_wine()
]

for data in data_list:
    x, y = data

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
        PowerTransformer(),
        PowerTransformer(method='yeo-johnson')
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
print('END')