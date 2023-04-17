import numpy as np
from sklearn.datasets import load_digits
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
x, y = load_digits(return_X_y=True)

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
allAlgorithms = all_estimators(type_filter='classifier')

max_r2 = 0
max_name = 'Failed'
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        results = model.score(x_test, y_test)
        print(name, 'SCORE:', results)

        if max_r2 < results:
            max_r2 = results
            max_name = name
    except:
        print(max_name, name)
print('==================================')
print('bestModelIs:', max_name, max_r2)

# AdaBoostClassifier SCORE: 0.20555555555555555
# BaggingClassifier SCORE: 0.9194444444444444
# BernoulliNB SCORE: 0.8833333333333333
# CalibratedClassifierCV SCORE: 0.9583333333333334
# CalibratedClassifierCV CategoricalNB
# CalibratedClassifierCV ClassifierChain
# CalibratedClassifierCV ComplementNB
# DecisionTreeClassifier SCORE: 0.8416666666666667
# DummyClassifier SCORE: 0.06111111111111111
# ExtraTreeClassifier SCORE: 0.8111111111111111
# ExtraTreesClassifier SCORE: 0.9861111111111112
# GaussianNB SCORE: 0.8416666666666667
