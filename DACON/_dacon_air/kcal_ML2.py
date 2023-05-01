import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, Normalizer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime
import pandas as pd
import numpy as np

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

X = train_df.drop('Calories_Burned', axis=1)
y = train_df['Calories_Burned']

from flaml import AutoML

MODEL_TIME_BUDGET = 60*5
MODEL_METRIC = 'mae'
MODEL_TASK = "regression"

auto_model = AutoML()
params = {
    "time_budget": MODEL_TIME_BUDGET,
    "metric": MODEL_METRIC,
    "task": MODEL_TASK,
    "seed": 42
}
auto_model.fit(X, y, **params)

print('Best hyperparmeter:', auto_model.model.estimator)
print('Best hyperparmeter config:', auto_model.best_config)
