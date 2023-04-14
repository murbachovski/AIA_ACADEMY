import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.svm import LinearSVC
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

warnings.filterwarnings(action='ignore')

#1. DATA
data_list = [
    load_iris(return_X_y=True),
    load_breast_cancer(return_X_y=True),
    load_wine(return_X_y=True),
]

model_list = [
    LinearSVC(),
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier()
]

data_name_list = [
    'iris',
    'cancer',
    'wine'
]

model_name_list = [
    'LinearSVC',
    'LogisticRegression',
    'DecisionTreeClassifier',
    'RandomForestClassifier',
]

#2. MODEL
for i, value in enumerate(data_list):
    x, y = value
    # print(x.shape, y.shape)
    print('==========================================')
    print(data_name_list[i])

    for j, value2 in enumerate(model_list):
        model = value2
        #3. COMPILE
        model.fit(x,y)
        #4. PREDICT
        results = model.score(x,y)
        print(model_name_list[j], 'model.score: ', results)
        y_predict = model.predict(x)
        acc = accuracy_score(y, y_predict)
        print(model_name_list[j], 'accuracy_score : ', acc)


# ==========================================
# iris
# LinearSVC 0.9666666666666667
# LogisticRegression 0.9733333333333334
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# ==========================================
# cancer
# LinearSVC 0.875219683655536
# LogisticRegression 0.9472759226713533
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# ==========================================
# wine
# LinearSVC 0.6179775280898876
# LogisticRegression 0.9662921348314607
# DecisionTreeClassifier 1.0
# RandomForestClassifier 1.0
# 회귀 R2
# 분류 ACC
