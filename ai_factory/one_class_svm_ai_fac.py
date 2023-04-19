import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# Load train and test data
path = './_data/ai_factory/'
save_path = './_save/ai_factory/'
train_data = pd.read_csv(path + 'train_data.csv')
test_data = pd.read_csv(path + 'test_data.csv')
submission = pd.read_csv(path + 'answer_sample.csv')

# Convert type column to HP column
def type_to_HP(type):
    HP = [30, 20, 10, 50, 30, 30, 30, 30]
    gen = (HP[i] for i in type)
    return list(gen)

train_data['type'] = type_to_HP(train_data['type'])
test_data['type'] = type_to_HP(test_data['type'])

# Select features
features = ['air_inflow', 'air_end_temp', 'out_pressure', 'motor_current', 'motor_rpm', 'motor_temp', 'motor_vibe']

# Prepare train and test data
X = train_data[features]
pca = PCA(n_components=7)
X = pca.fit_transform(X)
X_train, X_val = train_test_split(X, train_size=0.9, random_state=337)

# Normalize train and test data
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_val_normalized = scaler.transform(X_val)
test_data_normalized = scaler.transform(test_data[features])

#
from sklearn.metrics import f1_score

def lof_scoring(estimator, X, y):
    y_pred = estimator.fit_predict(X)
    return f1_score(y, y_pred, pos_label=-1)

# Tune Local Outlier Factor model
n_neighbors_range = range(10, 90, 2)
contamination_range = [0.04, 0.045, 0.05, 0.055, 0.06]
parameters = {'n_neighbors': n_neighbors_range, 'contamination': contamination_range}
lof = LocalOutlierFactor(leaf_size=99, algorithm='auto', metric='chebyshev', metric_params=None, novelty=False, p=3)
grid_search = GridSearchCV(estimator=lof, param_grid=parameters, cv=5, n_jobs=-1, verbose=1, scoring=lof_scoring)
grid_search.fit(X_train_normalized)
print('Best parameters: ', grid_search.best_params_)

# Predict anomalies in validation data
lof_best = grid_search.best_estimator_
y_pred_train = lof_best.fit_predict(X_train_normalized)
y_pred_val = lof_best.fit_predict(X_val_normalized)

# Predict anomalies in test data
y_pred_test = lof_best.fit_predict(test_data_normalized)
lof_predictions = [1 if x == -1 else 0 for x in y_pred_test]

# Save predictions to submission file
submission['label'] = pd.DataFrame({'Prediction': lof_predictions})
date = datetime.datetime.now().strftime("%m%d_%H%M")
submission.to_csv(save_path + date + '_LOF_submission.csv', index=False)
