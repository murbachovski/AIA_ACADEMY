import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor #투표

#3대장.
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor #연산할 필요 없는 것들을 빼버림, 잘나오는 곳 한쪽으로만 감.
from catboost import CatBoostRegressor


#1. 데이터
x,y  = load_breast_cancer(return_X_y=True)
x,y  = load_iris(return_X_y=True)
x,y  = load_wine(return_X_y=True)
x,y  = load_wine(return_X_y=True)

datasets = [
    load_breast_cancer(return_X_y=True),
    load_iris(return_X_y=True),
    load_wine(return_X_y=True),
    load_wine(return_X_y=True)
]

for i in datasets:
    x, y = i
    #2. 모델
    xgb = XGBRegressor()
    lg = LGBMRegressor()
    cat = CatBoostRegressor(verbose=0) #verbose 디폴트 1 

    model = VotingRegressor(
        estimators=[('XGB', xgb), ('LG', lg), ('CAT', cat)],
                    #voting='soft', #디폴트는 하드, 성능은 소프트가 더 좋음.
    )
    #3. 훈련
    model.fit(x, y)
    #4. 평가, 예측
    print()
    y_pred = model.predict(x)
    print('model.score : ', model.score(x,y))
    print("Voting.acc : ", r2_score(y,y_pred))
    Regressors = [xgb, lg, cat]
    li = []
    for model2 in Regressors:
        model2.fit(x, y)
        
        # 모델 예측
        y_predict = model2.predict(x)
        
        # 모델 성능 평가
        score2 = r2_score(y,y_predict)
        
        class_name = model2.__class__.__name__ 
    print("{0} R2 : {1:4f}".format(class_name, score2))
    li.append(score2)

# 리스트 출력
print(li)