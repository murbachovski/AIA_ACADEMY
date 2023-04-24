import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime
import pandas as pd
import numpy as np
import random
import os
from supervised.automl import AutoML
from sklearn.metrics import mean_squared_error

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# CALL DATA
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')


# Weight_Status, Gender => NUMBER
train_df['Weight_Status'] = train_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_df['Gender'] = train_df['Gender'].map({'M': 0, 'F': 1})
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})

train_df = train_df.drop('ID',  axis=1)
test_df = test_df.drop('ID', axis=1)

# PolynomialFeatures DATA 
# poly = PolynomialFeatures(degree=2, include_bias=False)
x = train_df.drop('Calories_Burned', axis=1)
y = train_df['Calories_Burned']

# SCALER
scaler = StandardScaler()
x = scaler.fit_transform(x)

# train, valid SPLIT
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

automl = AutoML(mode="Compete",
                eval_metric='rmse')
automl.fit(x, y)

# test PREDICT
pred = automl.predict(x)

#
rmse = RMSE(y, pred)
print('GPR RMSE : ', rmse)

# SUBMIT
sample_submission_df['Calories_Burned'] = pred
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
sample_submission_df.to_csv(save_path + date +"automl_fixed.csv", index=False)

# n_estimators: 앙상블 모델에서 사용할 결정트리의 개수
# max_depth: 결정트리에서 최대 깊이
# learning_rate: 경사하강법에서 사용할 학습률
# hidden_layer_sizes: 인공신경망 모델에서 은닉층의 크기
# activation: 인공신경망 모델에서 활성화 함수 선택
# dropout: 인공신경망 모델에서 드롭아웃 비율
# 데이터 전처리 관련 파라미터
# imputation_strategy: 결측치 처리 방법
# scaling_strategy: 데이터 스케일링 방법
# feature_selection_strategy: 특징 선택 방법
# categorical_encoding_strategy: 범주형 데이터 인코딩 방법
# 학습 관련 파라미터
# cv: 교차 검증(Cross-validation) 폴드 수
# scoring_metric: 모델 성능 측정 지표
# random_state: 랜덤 시드 값
# n_jobs: 병렬 처리에 사용할 CPU 개수