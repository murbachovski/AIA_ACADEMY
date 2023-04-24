import numpy as np
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
from sklearn.preprocessing import PolynomialFeatures

# Load the numpy array from the file
validation_indices = np.load('AutoML_04/split_validation_indices.npy')

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
poly = PolynomialFeatures(degree=2, include_bias=False)
x = poly.fit_transform(train_df.drop('Calories_Burned', axis=1))
y = train_df['Calories_Burned']

# SCALER
scaler = StandardScaler()
x = scaler.fit_transform(x)

# train, valid SPLIT
X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=42)
x_test = x.iloc[validation_indices]
y_test = y.iloc[validation_indices]
X_valid = x[validation_indices]
y_valid = y[validation_indices]

mlp = MLPRegressor(hidden_layer_sizes=(2000, 600,3),
                   max_iter=750, # = epochs = 반복 횟수
                   activation='relu',
                   solver='lbfgs', # 모델 학습에 사용되는 최적화 알고리즘
                   random_state=42,
                   verbose=1,
                   alpha=2,
                   #batch_size=20,
                   #tol=1e-6, # 성능향상에는 직접적으로 연관은 없겠구나 
                   #epsilon = 1e-8,
                   #early_stopping=True,
                   #shuffle=True
                   )
# Valid 데이터 RMSE: 0.340
# poly 뺀 값은 0.390
mlp.fit(X_train, y_train)

# Use the validation data to evaluate your machine learning model
y_pred = mlp.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred))
print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

# test PREDICT
X_test = test_df.values
X_poly_test = poly.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)
y_pred_test = mlp.predict(X_test_scaled)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

# SUBMIT
sample_submission_df['Calories_Burned'] = y_pred_test
sample_submission_df.to_csv(save_path + date + 'submission_MLP_Poly.csv', index=False)