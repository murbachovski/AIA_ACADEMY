import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.filterwarnings(action='ignore')
# Load train and test data
path='./_data/ai_factory/'
save_path= './_save/ai_factory/'
train_data = pd.read_csv(path+'train_data.csv')
test_data = pd.read_csv(path+'test_data.csv')
submission = pd.read_csv(path+'answer_sample.csv')

# 
def type_to_HP(type):
    HP=[30,20,10,50,30,30,30,30]
    gen=(HP[i] for i in type)
    return list(gen)
train_data['type']=type_to_HP(train_data['type'])
test_data['type']=type_to_HP(test_data['type'])
# print(train_data.columns)

# 
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]

# 
X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 3333)

# 
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

#
iso_forest = IsolationForest()
scoring = make_scorer(mean_squared_error, greater_is_better=False)
scoring_metric = 'neg_mean_squared_error'

param_grid = {
    'n_estimators': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                     21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                     31, 32, 33, 34, 35, 36, 37,  38, 39, 40,
                     41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                     51, 52, 53, 54, 55, 56, 57, 58, 59, 60
                     ],
    'contamination': [0.01, 0.02, 0.03, 0.04, 0.05,
                      0.06, 0.07, 0.08, 0.09
                      ],
    'max_samples': [10, 20, 30, 40, 50,
                    60, 70, 80, 90, 100
                    ]
}
grid_search = RandomizedSearchCV(iso_forest, param_grid, scoring=scoring, cv=5, refit=True)
grid_search.fit(X_train)
print("Best parameters: ", grid_search.best_params_)

# 
test_data_lof = scaler.fit_transform(test_data[features])
y_pred_test_lof = iso_forest.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")
submission.to_csv(save_path + date + '_iso_submission.csv', index=False)

print("최적의 매개변수 : ", grid_search.best_estimator_) 

print("최적의 파라미터 : ", grid_search.best_params_)

print("최적의 인덱스 : ", grid_search.best_index_)

print("BEST SCORE : ", grid_search.best_score_)
#0.9530071431