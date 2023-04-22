import pandas as pd
import numpy as np
import random
import os
from supervised.automl import AutoML
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import random
import os
import gc
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # 행운의 seed
train_df = pd.read_csv('/content/train.csv')
test_df = pd.read_csv('/content/test.csv')

# ID 열 제거
train_df = train_df.drop('ID', axis=1)
test_df = test_df.drop('ID', axis=1)

# Weight_Status, Gender 열을 숫자 데이터로 변환
train_df['Weight_Status'] = train_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_df['Gender'] = train_df['Gender'].map({'M': 0, 'F': 1})
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})

test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})

# PolynomialFeatures를 사용하여 데이터 전처리
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(train_df.drop('Calories_Burned', axis=1))
y = train_df['Calories_Burned']
test_df_poly = poly.fit_transform(test_df)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test = scaler.fit_transform(test_df_poly)

automl = AutoML(mode="Compete", eval_metric='rmse',total_time_limit=3600)

automl.fit(X,y)

preds = automl.predict(test_x)

y_train_orig = tmp_scaler.inverse_transform(preds.reshape(-1, 1))

original_predictions = np.expm1(preds)

original_y_pred = minmax_scaler.inverse_transform(preds.reshape(-1, 1))

original_predictions = np.expm1(original_y_pred)

import gc
gc.collect()
submission = pd.read_csv('/content/sample_submission.csv')
submission['Calories_Burned'] = y_train_orig
submission.to_csv('/content/scale.csv', index = False)