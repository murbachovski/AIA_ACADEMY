import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import numpy as np

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
print(X.shape)
pca = PCA(n_components=2)
X = pca.fit_transform(X)
print(X.shape)

# 
X_train, X_val = train_test_split(X, test_size= 0.3, random_state= 337337)
print(X_train.shape, X_val.shape)

#
pca = PCA(n_components=1)
X_train = pca.fit_transform(X_train)
X_val = pca.fit_transform(X_val)

# 
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data.iloc[:, :-1])
test_data_normalized = scaler.transform(test_data.iloc[:, :-1])

# 
n_neighbors = 42
contamination = 0.0452341899
lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                         contamination=contamination,
                         leaf_size=30,
                         algorithm='kd_tree',
                         metric='minkowski',
                         metric_params= None,
                         p=2
                         )
y_pred_train_tuned = lof.fit_predict(X_val)

# 
test_data_lof = scaler.fit_transform(test_data[features])
y_pred_test_lof = lof.fit_predict(test_data_lof)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test_lof]
#lof_predictions = [0 if x == -1 else 0 for x in y_pred_test_lof]

submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
print(submission.value_counts())

#time
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M")

submission.to_csv(save_path + date + '_REAL_LOF_submission.csv', index=False)

#0.9551928573
#0.9551928573
#0.9561993171
#0.9570394969
#0.9582241632