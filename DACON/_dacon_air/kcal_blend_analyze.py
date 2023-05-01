import pandas as pd
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder as le
from sklearn.ensemble import GradientBoostingRegressor as gbr, RandomForestRegressor as rfc
from xgboost import XGBRegressor as xgb
from lightgbm import LGBMRegressor as lgbm
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score as cv

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

train = pd.read_csv('./_data/dacon_kcal/train.csv')
test = pd.read_csv('./_data/dacon_kcal/test.csv')

train_x = train.drop(['ID','Calories_Burned'], axis=1)
train_y = train['Calories_Burned']

test_x = test.drop('ID', axis=1)
qual_col = []

for i in range(len(train_x.dtypes)):
    if (train_x.dtypes[i] == 'object'):
        qual_col.append(train_x.columns[i])

for col in qual_col:
    encoder = le()
    encoder.fit(train_x[col])
    train_x[col] = encoder.transform(train_x[col])
    
    for label in np.unique(test_x[col]):
        if label not in encoder.classes_:
            encoder.classes_ = np.append(encoder.classes_,label)
            
    test_x[col] = encoder.transform(test_x[col])

seed = 42

seed_everything(seed)
md_gbr = gbr(random_state=seed)
md_rfc = rfc(random_state=seed)
md_lgbm = lgbm(random_state=seed)
md_xgb = xgb(random_state=seed)
md_gbr.fit(train_x, train_y)
md_rfc.fit(train_x, train_y)
md_lgbm.fit(train_x, train_y)
md_xgb.fit(train_x, train_y)

pred_gbr = md_gbr.predict(test_x)
pred_rfc = md_rfc.predict(test_x)
pred_lgbm = md_lgbm.predict(test_x)
pred_xgb = md_xgb.predict(test_x)

pred_blend = (pred_rfc+pred_lgbm+pred_xgb)/3

submission = pd.read_csv('./_data/dacon_kcal/sample_submission.csv')

submission['Calories_Burned'] = pred_blend
submission.to_csv('./submit_blend.csv', index=False)

#################################################
mod_rfc = rfc(n_estimators=500, max_depth=20, max_features=None,  random_state=42)
mod_rfc.fit(train_x, train_y)

print('rfc_rmse')
pred_score = mod_rfc.predict(train_x)
print(mse(train_y,pred_score,squared=False))
# print()
print('rfc_cv')
cv_score = -cv(mod_rfc, train_x, train_y, scoring='neg_root_mean_squared_error' ,cv=5)
# print('cv rmse score')
# print(cv_score)

mod_lgbm = lgbm(n_estimators=450, num_leaves=17, max_depth=9, reg_sqrt=True,
                class_weight='balanced', reg_alpha=.15,
                objective='root_mean_squared_error', random_state=42)
mod_lgbm.fit(train_x, train_y)

print('lgbm_rmse')
pred_score = mod_lgbm.predict(train_x)
print(mse(train_y,pred_score,squared=False))
print('-'*20)
print('<lgbm_cv>')
cv_score = -cv(mod_lgbm, train_x, train_y, scoring='neg_root_mean_squared_error' ,cv=5)
print()
print('cv rmse score mean')
print(cv_score.mean())
print()
print('cv rmse score std')
print(cv_score.std())

submission = pd.read_csv('./_data/dacon_kcal/sample_submission.csv')

submission['Calories_Burned'] = pred_score
submission.to_csv('./_save/dacon_kcal/submit_blend_pred.csv', index=False)

