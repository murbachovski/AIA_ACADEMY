import numpy as np
import xgboost as xgb
import numpy as np
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import mean_squared_error
def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))

# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')
test_df = pd.read_csv(path + 'test.csv')
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})
test_df = test_df.drop('ID', axis=1)

model = xgb.Booster()

import xgboost as xgb
import numpy as np

# Load the model from the file
booster = xgb.Booster()
booster.load_model('./dacon/AutoML_04/16_Xgboost_Stacked/learner_fold_9.xgboost')

# Load new data
# new_data = np.loadtxt(test_df, delimiter=',')

# Generate predictions for the new data
predictions = booster.predict(xgb.DMatrix(test_df))

# SUBMIT
sample_submission_df['Calories_Burned'] = predictions
date = datetime.datetime.now()
date = date.strftime('%m%d_%H%M%S')
sample_submission_df.to_csv(save_path + date +"best_ML_LOAD_SUBMIT.csv", index=False)