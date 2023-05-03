from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings(action='ignore')
import time
from sklearn.metrics import mean_squared_error, accuracy_score
from catboost import CatBoostClassifier, CatBoostRegressor
# DACNO DDARUNG
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error # = MSE
import pandas as pd # 전처리(CSV -> 데이터화)

#1. DATA
path = './_data/dacon/dacon_ddarung/' # path ./은 현재 위치
path_save = './_save/dacon/dacon_ddarung/'
# Column = Header

# TRAIN
train_csv = pd.read_csv(path + "train.csv",
                        index_col=0) 
# print(train_csv)
# print(train_csv.shape) # (1459, 10)


# TEST
test_csv = pd.read_csv(path + "test.csv",
                        index_col=0) 
# print(test_csv)
# print(test_csv.shape) # (715, 9)
# print(train_csv.columns) #        'hour_bef_windspeed', 'hour_bef_humidity',    'hour_bef_visibility',
                         #        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
                         #       dtype='object')
# print(train_csv.info())

# ---  ------                  --------------  -----
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
# ---  ------                  --------------  -----

# print(train_csv.describe()) # [8 rows x 10 columns]

# print("TYPE",type[train_csv])

############################결측치 처리#############################
#1. 결측치 처리 - 제거
# print(train_csv.isnull().sum())  #중요하답니다.
train_csv = train_csv.dropna()
# print(train_csv.isnull().sum())
# print(train_csv.info())
# print(train_csv.shape) # (1328, 10)


#########################################train_csv데이터에서 x와 y를 분리했다.
#########이게 중요합니다#######
x = train_csv.drop(['count'], axis = 1)
# print(x)
y = train_csv['count']
# print(y)
#########################################train_csv데이터에서 x와 y를 분리했다.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

from hyperopt import  hp, fmin, Trials, STATUS_OK, tpe

# 2. MODEL
search_space = {
                'learning_rate' : hp.uniform('learning_rate', 0.001, 1),
                'depth' : hp.quniform('max_depth', 3, 16, 1),
                # 'num_leaves' : hp.quniform('num_leaves', 24, 64, 1),
                # 'min_child_samples' : hp.quniform('min_child_samples', 10, 200),
                # 'min_child_weight' : hp.quniform(1, 50, 1),
                # 'subsample' : hp.uniform('subsample', 0.5, 1),
                # 'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                # 'max_bin' : hp.quniform('max_bin', 10, 500, 1),
                # 'reg_lambda': hp.uniform('reg_lambda', 0.001, 10),
                # 'reg_alpha' : hp.uniform('reg_alpha', 0.001, 50)
}
# hp.quniform(labe, low, high, q)   : 최소부터 최대까지 q 간격
# hp.uniform(label, low, high)      : 최소부터 최대까지 정규분포 간격
# hp.randint(label, upper)          : # 0부터 최대값 upper까지 random한 정수값
# hp.loguniform(label, low, high)   :exp(uniform(low, hish))값 반환/ 이거 역시 정규분포

def lgb_hamsu(search_space):
    params = {
        'n_estimators' : 2,
        'learning_rate' : search_space['learning_rate'],
        'depth' : int(search_space['depth']), # 무적권 정수형
        # 'num_leaves' : int(search_space['num_leaves']),
        # 'min_child_samples' : int(round(min_child_samples)),
        # 'min_child_weight' : min_child_weight,
        # 'subsample' : search_space['subsample'], # Dropout과 비슷한 녀석 0~1 사이의 값
        # 'colsample_bytree' : colsample_bytree,
        # 'max_bin' : max(int(round(max_bin)), 10), # 무적권 10이상
        # 'reg_lambda': max(reg_lambda, 0), # 무적권 양수만 나오게 하기 위해서
        # 'reg_alpha' : reg_alpha
    }

    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_test, y_test)],
            # eval_metric='rmse',
            verbose=0,
            early_stopping_rounds=5
              )
    y_predict = model.predict(x_test)
    results = mean_squared_error(y_test, y_predict)
    return results

trial_val = Trials()    # hist를 보기위해

best = fmin(
    fn=lgb_hamsu,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=trial_val,
    rstate=np.random.default_rng(seed=10)
)

trial_val_all = pd.DataFrame(trial_val)
print(trial_val_all)
print(best)

# 41      2   41  None  {'loss': 2884.4646287287947, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:55.682 2023-05-03 09:51:56.244     
# 42      2   42  None  {'loss': 2635.2889072921694, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:56.266 2023-05-03 09:51:56.399     
# 43      2   43  None  {'loss': 2704.2982573959393, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:56.409 2023-05-03 09:51:56.484     
# 44      2   44  None  {'loss': 2355.8850802592447, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:56.488 2023-05-03 09:51:56.570     
# 45      2   45  None   {'loss': 3405.843872817739, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:56.587 2023-05-03 09:51:56.702     
# 46      2   46  None    {'loss': 2496.92514022183, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:56.722 2023-05-03 09:51:56.804     
# 47      2   47  None   {'loss': 7252.003479901239, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:56.818 2023-05-03 09:51:56.887     
# 48      2   48  None  {'loss': 2989.9137819814655, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:56.887 2023-05-03 09:51:57.011     
# 49      2   49  None  {'loss': 2306.6524486404155, 'status': 'ok'}  ...  None       0 2023-05-03 09:51:57.017 2023-05-03 09:51:57.103     

# [50 rows x 10 columns]
# {'learning_rate': 0.8640344838748092, 'max_depth': 6.0}