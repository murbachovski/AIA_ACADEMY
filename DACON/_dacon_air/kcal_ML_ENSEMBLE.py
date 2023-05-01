import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime
import pandas as pd
import numpy as np
from xgboost import XGBRFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer, Normalizer
import catboost
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import PolynomialFeatures

# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# Weight_Status, Gender => NUMBER
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submit_csv = pd.read_csv(path + 'sample_submission.csv')

# Weight_Status, Gender => NUMBER
train_csv['Weight_Status'] = train_csv['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_csv['Gender'] = train_csv['Gender'].map({'M': 0, 'F': 1})
test_csv['Weight_Status'] = test_csv['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_csv['Gender'] = test_csv['Gender'].map({'M': 0, 'F': 1})

poly = PolynomialFeatures(degree=2, include_bias=False)
x = poly.fit_transform(train_csv.drop(['Calories_Burned'], axis=1))
y = train_csv['Calories_Burned']

x_test = poly.fit_transform(test_csv)

# scaler = MaxAbsScaler()
# x = scaler.fit_transform(x)


RF1 = RandomForestRegressor(max_features=0.6, max_leaf_nodes=300,n_estimators=2, n_jobs=-1)
RF1.fit(x, y)

RF2 = RandomForestRegressor(max_features=1.0, max_leaf_nodes=400, n_estimators=3,n_jobs=-1)
RF2.fit(x, y)

ET = ExtraTreesRegressor(max_features=9, max_leaf_nodes=200, n_estimators=21, n_jobs=-1)
ET.fit(x, y)

RFX = XGBRFRegressor(random_state=42)
RFX.fit(x, y)

GPR = GaussianProcessRegressor(alpha=10, n_restarts_optimizer=300)

ensemble_result = (RF1.predict(x_test) +
                   RF2.predict(x_test) +
                   ET.predict(x_test)  +
                   RFX.predict(x_test) +
                   GPR.predict(x_test)
                   
                   ) / 5

# from flaml import AutoML

# MODEL_TIME_BUDGET = 60*5
# MODEL_METRIC = 'mae'
# MODEL_TASK = "regression"

# auto_model = AutoML()
# params = {
#     "time_budget": MODEL_TIME_BUDGET,
#     "metric": MODEL_METRIC,
#     "task": MODEL_TASK,
#     "seed": 42
# }
# auto_model.fit(X, y, **params)

# print('Best hyperparmeter:', auto_model.model.estimator)
# print('Best hyperparmeter config:', auto_model.best_config)

# valid PREDICT
# y_pred_valid = mlp.predict(X_valid)
# rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
# print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

rmse = RMSE(y, ensemble_result)
print('GPR RMSE : ', rmse)

file_name = 'ENSEMBLE_submit_final.csv'
submit_csv['Calories_Burned'] = ensemble_result
submit_csv.to_csv(save_path + date + file_name, index=False)

#각각 모델의 골든 파라미터 찾고 스케일러, PCA, poly 적용 여부 확인...!