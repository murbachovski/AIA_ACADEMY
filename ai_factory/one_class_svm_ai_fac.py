import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import OneClassSVM

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
X_train, X_val = train_test_split(X, train_size= 0.9, random_state= 7777)

# 
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# 
kernel = 'sigmoid'
degree = 3
gamma = 'auto'
coef0 = 0.1
tol = 1e-3
nu = 0.5
shrinking = True
cache_size = 200
verbose = 1
max_iter = -1
ocsvm = OneClassSVM(kernel= kernel,
                     nu=nu,
                     gamma=gamma,
                     degree = degree,
                     coef0 = coef0,
                     tol = tol,
                     shrinking = shrinking ,
                     cache_size = cache_size,
                     verbose = verbose,
                     max_iter = 1
                    )
y_pred_train_tuned = ocsvm.fit_predict(X_val)

# 
test_data_ocsvm = scaler.fit_transform(test_data[features])
y_pred_test_ocsvm = ocsvm.predict(test_data_ocsvm)
ocsvm_predictions = [1 if x == -1 else 0 for x in y_pred_test_ocsvm]
#ocsvm_predictions = [0 if x == -1 else 1 for x in y_pred_test_ocsvm]


submission['label'] = pd.DataFrame({'Prediction': ocsvm_predictions})
print(submission.value_counts())
#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + '_REAL_OCSVM_submission.csv', index=False)

#0.9628766067
