import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler는 이상치도 잡아준다
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd
x, y = load_diabetes(return_X_y=True)



# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

parameters = {
            'n_estimators' : 1000, # epochs
            'learning_rate' : 0.3, # 
            'max_depth' : 2, #
            'gamma' : 0, #
            'min_child_weight' : 0, #
            'subsample' : 0.2, # 
            'colsample_bytree' : 0.5,
            'colsample_bylevel' : 0,
            'colsample_bynode' : 0,
            'reg_alpha' : 1,
            'reg_lambda' : 1,
            'random_state' : 337
}

# 2. MODEL
model = XGBRegressor(**parameters)

for i in model.feature_importances_:
	min = min(i)
	print(min)
	x = pd.DataFrame(x)
	for j in x.columns:
		print(x.shape)
		x = x.drop(x.columns[j], axis=1)
		print(x.shape)
		x_train, x_test, y_train, y_test = train_test_split(
			x,
			y,
			random_state=337,
			shuffle=True,
			test_size=0.2,
			# stratify=y
    )
	# 3. COMPILE
	model.fit(x_train, y_train,
			eval_set=[(x_train, y_train), (x_test, y_test)],
			early_stopping_rounds=10,
			verbose=True,
			#   eval_metric='logloss' # 이진분류
			# eval_metric='auc' # 이진분류
			#   eval_metric='error' # 이진분류
			#   eval_metric='merror' # 다중분류
			# eval_metric='merror', # 'mae, 'rmsle # 회귀 
				eval_metric='rmse',  # 회귀 
			)
	# print(x_train.columns)
	results = model.score(x_test, y_test)
	print('FINISH SCORE: ', results)

	y_predict = model.predict(x_test)

	# acc = accuracy_score(y_test, y_predict)
	# print('acc: ', acc)

	mse = mean_squared_error(y_test, y_predict)
	print("RMSE: ", np.sqrt(mse))
	print('한번 끝')

##############################################################################
    # print(model.feature_importances_)