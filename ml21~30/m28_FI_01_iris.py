# [실습]
# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 재구성후
# 모델을 돌려서 결과 도출

# 기존 모델들과 성능비교
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, load_wine
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 결과 비교
datasets = [
    ("Iris", load_iris()),
    # ("Breast Cancer", load_breast_cancer()),
    # ("Diabetes", load_diabetes()),
    ("Digits", load_digits()),
    ("Wine", load_wine())
]

model_list = [
    ("DTC", DecisionTreeClassifier()),
    ("RFC", RandomForestClassifier()),
    ("GBC", GradientBoostingClassifier()),
    ("XGB", XGBClassifier())
]

for dataset_name, dataset in datasets:
    x, y = dataset.data, dataset.target
    print(f"Dataset: {dataset_name}")
    for model_name, model in model_list:
        model.fit(x, y)
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        print(model_name)
        print('기존 ACC : ', accuracy)
        print('model_feature: ', model.feature_importances_)
        for i in model.feature_importances_:
            if i < 0.05:
                datasets.drop(i, axis=1)
    print('FINISH')

