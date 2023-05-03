import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.linear_model import Lasso, Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.metrics import mean_absolute_error

x, y = load_linnerud(return_X_y=True)

# print(x)
# print(y)
# print(x.shape) # (20, 3)
# print(y.shape) # (20, 3)

# [2. 110. 43] => [138. 33. 68.]
# model = Ridge()
# model.fit(x, y)
# y_pred = model.predict(x)
# print("SCORE: ", np.rounde(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[2, 110, 43]]))

# model = XGBRegressor()
# model.fit(x, y)
# y_pred = model.predict(x)
# print("SCORE: ", mean_absolute_error(y, y_pred))
# print(model.predict([[2, 110, 43]]))

# model = LGBMRegressor() # ERROR
# model.fit(x, y)
# y_pred = model.predict(x)
# print("SCORE: ", mean_absolute_error(y, y_pred))
# print(model.predict([[2, 110, 43]]))

# model = MultiOutputRegressor(LGBMRegressor())
# model.fit(x, y)
# y_pred = model.predict(x)
# print("SCORE: ", mean_absolute_error(y, y_pred))
# print(model.predict([[2, 110, 43]]))
###################################################################################################
# model = CatBoostRegressor() # ERROR
# model.fit(x, y)
# y_pred = model.predict(x)
# print("SCORE: ", np.rounde(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[2, 110, 43]])) 

# model = MultiOutputRegressor(CatBoostRegressor()) # ERROR
# model.fit(x, y)
# y_pred = model.predict(x)
# print("SCORE: ", np.round(mean_absolute_error(y, y_pred), 4))
# print(model.predict([[2, 110, 43]]))

model = CatBoostRegressor(loss_function='MultiRMSE')
model.fit(x, y)
y_pred = model.predict(x)
print("SCORE: ", np.round(mean_absolute_error(y, y_pred), 4))
print(model.predict([[2, 110, 43]])) 

