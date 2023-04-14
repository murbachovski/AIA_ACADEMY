import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import all_estimators
import sklearn as sk
print(sk.__version__)
#1. DATA
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    random_state=222,
    shuffle=True,
    test_size=0.2
)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. MODEL
# model = RandomForestRegressor(n_jobs=4)
allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='classifier')

# print('allAlgorithms: ', allAlgorithms)
# print('model_numbers: ', len(allAlgorithms))

for (name, algorithm) in allAlgorithms:
    model = algorithm()
    #3. COMPILE
    model.fit(x_train, y_train)

    #4. PREDICT
    results = model.score(x_test, y_test) # r2_score
    print('model.score: ', results)
    print(np.dtype(y_test))
    print(np.dtype(y_predict))
    y_predict = model.predict(x_test)
    r2_score = r2_score(y_test, y_predict)
    print('r2_score: ', r2_score)