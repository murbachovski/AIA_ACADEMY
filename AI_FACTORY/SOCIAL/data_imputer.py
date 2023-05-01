import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
import pandas as pd

x = pd.read_csv('./_data/ai_factory/social/train_all.csv')

imputer = IterativeImputer(estimator=XGBRegressor(
                       tree_method='gpu_hist',
                       predictor='gpu_predictor',
                       gpu_id=0,n_estimators=100,learning_rate=0.3,
                       max_depth=6
))

x.columns[4] = imputer.fit_transform(x.columns[4])
print(x.columns[4])