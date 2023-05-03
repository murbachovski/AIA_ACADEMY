import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler는 이상치도 잡아준다
from xgboost import XGBClassifier, XGBRegressor


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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = {
            'n_estimators' : [100], # epochs
            'learning_rate' : [0.5], # 
            'max_depth' : [2], #
            'gamma' : [0], #
            'min_child_weight' : [0.5], #
            'subsample' : [0.7], # 
            'colsample_bytree' : [0.7],
            'colsample_bytree1' : [0],
            'colsample_bynode' : [0.2],
            'reg_alpha' : [0.01],
            'reg_lamda' : [0]
}

# 2. MODEL
model = XGBClassifier(random_state = 337)

# 3. COMPILE
model.fit(x_train, y_train,
          eval_set=[(x_test, y_test)],
          early_stopping_rounds=10
          )

# 4. PREDICT
# print('BEST PARAMETERS: ', model.best_params_)
# print('BEST SCORE: ', model.best_score_)

results = model.score(x_test, y_test)
print('FINISH SCORE: ', results)

# BEST PARAMETERS:  {'n_estimators': 100}
# BEST SCORE:  0.9648351648351647
# FINISH SCORE:  0.9473684210526315

# BEST PARAMETERS:  {'learning_rate': 0.5}
# BEST SCORE:  0.9714285714285713
# FINISH SCORE:  0.9473684210526315

# BEST PARAMETERS:  {'max_depth': 2}
# BEST SCORE:  0.9692307692307691
# FINISH SCORE:  0.956140350877193

# BEST PARAMETERS:  {'gamma': 0}
# BEST SCORE:  0.9648351648351647
# FINISH SCORE:  0.9473684210526315

# BEST PARAMETERS:  {'min_child_weight': 0.5}
# BEST SCORE:  0.9670329670329669
# FINISH SCORE:  0.9385964912280702

# BEST PARAMETERS:  {'subsample': 0.7}
# BEST SCORE:  0.9714285714285715
# FINISH SCORE:  0.9385964912280702

# BEST PARAMETERS:  {'colsample_bytree': 0.7}
# BEST SCORE:  0.9670329670329669
# FINISH SCORE:  0.9385964912280702

# BEST PARAMETERS:  {'colsample_bytree1': 0}
# BEST SCORE:  0.9648351648351647
# FINISH SCORE:  0.9473684210526315

# BEST PARAMETERS:  {'colsample_bynode': 0.2}
# BEST SCORE:  0.9692307692307691
# FINISH SCORE:  0.9473684210526315

# BEST PARAMETERS:  {'reg_alpha': 0.01}
# BEST SCORE:  0.9670329670329669
# FINISH SCORE:  0.9385964912280702

# BEST PARAMETERS:  {'reg_lamda': 0}
# BEST SCORE:  0.9648351648351647
# FINISH SCORE:  0.9473684210526315

# BEST PARAMETERS:  {'colsample_bynode': 0.2, 'colsample_bytree': 0.7, 'colsample_bytree1': 0, 'gamma': 0, 'learning_rate': 0.5, 'max_depth': 2, 'min_child_weight': 0.5, 'n_estimators': 100, 'reg_alpha': 0.01, 'reg_lamda': 0, 'subsample': 0.7}
# BEST SCORE:  0.9692307692307691
# FINISH SCORE:  0.956140350877193