# 오늘 배운 결측치 처리 자유롭게 활용
# 성능 올려봐!!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import null
from tensorflow.python.keras.models import Sequential,load_model, Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm_notebook
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import time
import datetime
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

imputer_switch = True

#1. 데이터
path = './_data/kaggle_house/' 

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

# drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
# test_csv.drop(drop_cols, axis = 1, inplace =True)


x = train_csv.drop(['SalePrice'], axis = 1)
y = train_csv['SalePrice']

if imputer_switch == True:
    imputer = IterativeImputer(estimator=XGBRegressor())
    x = imputer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)
model = XGBRegressor()
model.fit(x_train, y_train)

results = model.score(x_test, y_test)
print('results: ', results)

# 결측치 처리 전
# results:  0.7930421636665577

# 결측치 처리 후
# results:  0.7927142976659465