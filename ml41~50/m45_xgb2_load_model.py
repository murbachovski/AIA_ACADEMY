import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler는 이상치도 잡아준다
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score

x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=337,
    shuffle=True,
    test_size=0.2,
    stratify=y
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

# parameters = {
#             'n_estimators' : 100, # epochs
#             'learning_rate' : 5, # 
#             'max_depth' : 2, #
#             'gamma' : 0, #
#             'min_child_weight' : 1, #
#             'subsample' : 1, # 
#             'colsample_bytree' : 1,
#             'colsample_bylevel' : 1,
#             'colsample_bynode' : 1,
#             'reg_alpha' : 0,
#             'reg_lambda' : 1,
#             'random_state' : 337
# }

# # 2. MODEL
# model = XGBClassifier(**parameters)

# # 3. COMPILE
# hist = model.fit(x_train, y_train,
#           eval_set=[(x_train, y_train), (x_test, y_test)],
#           early_stopping_rounds=10,
#           verbose=True,
#         #   eval_metric='logloss' # 이진분류
#           eval_metric='auc' # 이진분류
#         #   eval_metric='error' # 이진분류
#         #   eval_metric='merror' # 다중분류
#         #   eval_metric='merror', 'mae, 'rmsle # 회귀 
#           )

path = './_temp/'
# from tensorflow.keras.models import load_model

model = XGBClassifier()
model.load_model(path + 'm45_xgb1_save_model.dat')

results = model.score(x_test, y_test)
print('FINISH SCORE: ', results)

y_predict = model.predict(x_test)


acc = accuracy_score(y_test, y_predict)
print('acc: ', acc)

##############################################################################
# # import pickle
# import joblib
# joblib.dump(model, path + 'm43_joblib1_save.dat')
