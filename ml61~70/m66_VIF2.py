from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

# 1. DATA
datasets = fetch_california_housing()
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df['target'] = datasets.target


y = df['target']
x = df.drop(['target'], axis=1)

# 다중공선성
vif = pd.DataFrame()
vif['variables'] = x.columns

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

vif['VIF'] = [variance_inflation_factor(x_scaled, i) for i in range(x_scaled.shape[1])]
# print(vif)
#     variables       VIF
# 0      MedInc  2.501295
# 1    HouseAge  1.241254
# 2    AveRooms  8.342786
# 3   AveBedrms  6.994995
# 4  Population  1.138125
# 5    AveOccup  1.008324
# 6    Latitude  9.297624
# 7   Longitude  8.962263

x = x.drop(['Latitude'], axis=1)
x = x.drop(['Latitude', 'Logitude'], axis=1)

print(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, shuffle=True, random_state=337, test_size=0.2, # stratify=y
)

scaler2 = StandardScaler()
x_scaled2 = scaler2.fit_transform(x_train)
x_test = scaler2.transform(x_test)

#2. MODEL
model = RandomForestRegressor(random_state=337)

#3. COMPILE
model.fit(x_train, y_train)

#4. PREDICT
results = model.score(x_test, y_test)
print('results', results)