import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler # RobustScaler는 이상치도 잡아준다
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=337,
    shuffle=True,
    test_size=0.2,
    # stratify=y
)

# scaler = RobustScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=337)

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

# 3. COMPILE
hist = model.fit(x_train, y_train,
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

results = model.score(x_test, y_test)
print('FINISH SCORE: ', results)

y_predict = model.predict(x_test)


# acc = accuracy_score(y_test, y_predict)
# print('acc: ', acc)

mse = mean_squared_error(y_test, y_predict)
print("RMSE: ", np.sqrt(mse))


##############################################################################
# print(model.feature_importances_)
# [0.1204448  0.03181682 0.07185043 0.10543638 0.08690013 0.0775187
#  0.14735512 0.06484857 0.13835293 0.15547612]
threshold = np.sort(model.feature_importances_)
print(threshold)
from sklearn.feature_selection import SelectFromModel

for i in threshold:
    selection = SelectFromModel(model, threshold=i, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = LinearRegression()
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    print("Tresh=%.3f, n=%d, R2: %.2f%%" %(i, select_x_train.shape[1], score* 100))
