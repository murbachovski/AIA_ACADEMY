import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, mean_squared_error, f1_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
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

# Define the hyperparameters for the LOF algorithm
lof_param_grid = {
    'n_neighbors': [5, 10, 20, 30, 31, 32, 33, 34, 37, 38, 39, 40, 50],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [10, 20, 30, 40, 50],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'contamination': [0.01, 0.02, 0.03, 0.04, 0.05]
}

# Create an instance of the LOF estimator with default hyperparameters
lof = LocalOutlierFactor()

#
scoring = make_scorer(f1_score)

# Create an instance of GridSearchCV to tune the hyperparameters of the LOF estimator
lof_cv = HalvingGridSearchCV(estimator=lof, param_grid=lof_param_grid, cv=5, scoring=scoring, verbose=1)

# Fit the LOF estimator on the training data
lof_cv.fit(X_train)

# Get the best hyperparameters
best_lof_params = lof_cv.best_params_

# Create an instance of the LOF estimator with the best hyperparameters
lof = LocalOutlierFactor(**best_lof_params)

# Fit the LOF estimator on the training data
lof.fit(X_train)

# Use the LOF estimator to predict the outliers in the test data
y_pred_test_lof = lof.fit_predict(test_data[features])
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]

# Add the predictions to the submission file
submission['label'] = pd.DataFrame({'Prediction': lof_predictions})

# Save the submission file
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + '_PARAM_LOF_submission.csv', index=False)

print("최적의 매개변수 : ", lof_cv.best_estimator_) 

print("최적의 파라미터 : ", lof_cv.best_params_)

print("최적의 인덱스 : ", lof_cv.best_index_)

print("BEST SCORE : ", lof_cv.best_score_)

#0.9530071431