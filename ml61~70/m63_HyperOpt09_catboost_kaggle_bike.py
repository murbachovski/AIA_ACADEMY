from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer, load_wine, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings(action='ignore')
import time
from sklearn.metrics import mean_squared_error, accuracy_score
from catboost import CatBoostClassifier, CatBoostRegressor

#1. PATH
path = ('./_data/kaggle/kaggle_bike/')
path_save = ('./_save/kaggle/kaggle_bike/')

#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv', index_col=0) #index_col이 필요한 상황일까?
print(train_csv.shape)

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
print(test_csv.shape)
print(train_csv.columns)
print(test_csv.columns)

#1-4. ISNULL(결측치 처리)
print(train_csv.isnull().sum()) # 결측치가 다 없네?(꽉 차 있는 상황인가?)
train_csv = train_csv.dropna()

#1-5. DROP(x, y데이터 분리)
x = train_csv.drop(['count', 'casual', 'registered'], axis=1) # 이곳에서 데이터를 같이 제거해준다.
y = train_csv['count']
print('x.columns: ', x.columns)
print(train_csv.columns)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8
)

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

# 41      2   41  None  {'loss': 23297.408232595553, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:34.438 2023-05-03 10:08:34.520     
# 42      2   42  None  {'loss': 23730.626107291802, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:34.531 2023-05-03 10:08:34.620     
# 43      2   43  None   {'loss': 23156.45121267511, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:34.620 2023-05-03 10:08:34.737     
# 44      2   44  None   {'loss': 23561.44583461195, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:34.742 2023-05-03 10:08:34.852     
# 45      2   45  None  {'loss': 25017.689542329426, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:34.852 2023-05-03 10:08:34.950     
# 46      2   46  None  {'loss': 24527.465171226286, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:34.956 2023-05-03 10:08:35.030     
# 47      2   47  None   {'loss': 25133.13867438374, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:35.036 2023-05-03 10:08:35.118     
# 48      2   48  None  {'loss': 23158.257392182637, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:35.126 2023-05-03 10:08:35.203     
# 49      2   49  None  {'loss': 23609.358926442088, 'status': 'ok'}  ...  None       0 2023-05-03 10:08:35.219 2023-05-03 10:08:35.364     

# [50 rows x 10 columns]
# {'learning_rate': 0.7959347005368931, 'max_depth': 12.0}