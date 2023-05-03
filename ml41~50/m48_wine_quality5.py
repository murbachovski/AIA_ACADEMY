import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler
from sklearn.preprocessing import StandardScaler,RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

# Load data
path = './_data/wine/'
path_save = './_save/wine/'
train_csv = pd.read_csv(path + 'train.csv', index_col= 0)
test_csv = pd.read_csv(path + 'test.csv', index_col= 0)

# Remove rows with single class label
single_class_label = train_csv['quality'].nunique() == 1
print(single_class_label)
if single_class_label:
    train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# Label encode 'type'
le = LabelEncoder()
train_csv['type'] = le.fit_transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# Split data
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
class_numbers = np.unique(y, return_counts=True)       # Get the unique values in y


x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=850, train_size=0.7, stratify=y)

# Scale data
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

model = RandomForestClassifier()

model.fit(x_train, y_train) 

# Evaluate model
model.fit(x_test, y_test)

y_predict = model.predict(x_test)
results = model.score(x_test, y_test)

acc = accuracy_score(y_test, y_predict)
print("최종점수:", results)
print("acc 는", acc)

# [실습] y의 클래스를 7개에서 5개로 줄여서 성능을 비교!!
