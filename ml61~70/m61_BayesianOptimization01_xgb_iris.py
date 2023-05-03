from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor, LGBMClassifier
import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings(action='ignore')
import time
from xgboost import XGBClassifier

# 1. DATA
x, y = load_iris(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=337, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. MODEL
# bayesian_params = {
#                 'learning_rate' : (0.001, 1),
#                 'max_depth' : (3, 16),
#                 'num_leaves' : (24, 64),
#                 'min_child_samples' : (10, 200),
#                 'min_child_weight' : (1, 50),
#                 'subsample' : (0.5, 1),
#                 'colsample_bytree' : (0.5, 1),
#                 'max_bin' : (10, 500),
#                 'reg_lambda': (0.001, 10),
#                 'reg_alpha' : (0.01, 50)
# }

bayesian_params = {
                'learning_rate' : (0.001, 1),
}

def lgb_hamsu(learning_rate):
    params = {
        'learning_rate' : learning_rate,
    }

    model = XGBClassifier(**params)
    model.fit(x_train, y_train, 
            eval_metric='rmse',
            verbose=1,
              )
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=337,
                              
                              )
start_time = time.time()
lgb_bo.maximize(init_points=5, n_iter=100)
end_time = time.time()
print(end_time - start_time, '초')
print(lgb_bo.max)
# 12.085485458374023 초
# {'target': 0.9424184261036468, 'params': {'learning_rate': 0.8001068868516572}}