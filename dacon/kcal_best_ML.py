import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from supervised.automl import AutoML # mljar-supervised
import time
import datetime

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# CALL DATA
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')

# DATA 
X = train_df.drop('Calories_Burned', axis=1)
y = train_df['Calories_Burned']
test = test_df

# train, valid SPLIT
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# train models with AutoML
automl = AutoML(mode="Compete", eval_metric='rmse', n_jobs=-1)
automl.fit(X, y)


# compute the MSE on test data
predictions = automl.predict(test)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

rmse = RMSE(y, predictions)
print('GPR RMSE : ', rmse)

sample_submission_df['Calories_Burned'] = predictions
sample_submission_df.to_csv(date + "automl_best_fixed.csv", index=False)
# Ensemble mae 0.808298 trained in 0.33 seconds
# AutoML fit time: 58.45 seconds
# AutoML best model: Ensemble
# Test MSE: 0.8464317096259125

# 기본으로 넣어주면 성능이 별로인데...
# 그렇다면 스케일러...? PCA...? nomal...? drop...?