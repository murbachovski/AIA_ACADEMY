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

# 1. DATA
x, y = load_digits(return_X_y=True)

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

# 41      2   41  None   {'loss': 3.217063685009555, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:07.932 2023-05-03 09:49:08.352     
# 42      2   42  None  {'loss': 2.5423553168070727, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:08.367 2023-05-03 09:49:08.541     
# 43      2   43  None   {'loss': 4.540707849882959, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:08.547 2023-05-03 09:49:08.667     
# 44      2   44  None     {'loss': 4.3581442020788, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:08.687 2023-05-03 09:49:08.819     
# 45      2   45  None  {'loss': 2.8985939457422853, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:08.826 2023-05-03 09:49:08.966     
# 46      2   46  None   {'loss': 8.221873788870946, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:08.971 2023-05-03 09:49:09.119     
# 47      2   47  None  {'loss': 2.9360797549058804, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:09.125 2023-05-03 09:49:09.283     
# 48      2   48  None   {'loss': 2.745373927499832, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:09.289 2023-05-03 09:49:09.432     
# 49      2   49  None  {'loss': 3.8868934437763962, 'status': 'ok'}  ...  None       0 2023-05-03 09:49:09.443 2023-05-03 09:49:09.568     

# [50 rows x 10 columns]
# {'learning_rate': 0.9975378530878217, 'max_depth': 11.0}