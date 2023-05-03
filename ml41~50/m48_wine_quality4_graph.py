# 그래프 그린다
# 1. do not use value_counts
# 2. do not use np.unique.return_counts
# 3. Use groupby, count

######### You can get plt.bar (quality column)

# hint
# 데이터 갯수(y축) = 데이터갯수.주저리 

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
# print(single_class_label)
if single_class_label:
    train_csv = train_csv[train_csv['quality'] != train_csv['quality'].unique()[0]]

# Label encode 'type'
le = LabelEncoder()
train_csv['type'] = le.fit_transform(train_csv['type'])
test_csv['type'] = le.transform(test_csv['type'])

# Split data
x = train_csv.drop(['quality'], axis=1)
y = train_csv['quality']
import matplotlib.pyplot as plt
y = y.groupby('quality')
print(y.first())

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=850, train_size=0.7, stratify=y)

# # One-hot encode 'y'
# y_train = pd.get_dummies(y_train)
# y_test = pd.get_dummies(y_test)

# Scale data
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test_csv = scaler.transform(test_csv)

model = RandomForestClassifier()
# model.set_params(#**parameters, 
# **params
#                  )
model.fit(x_train, y_train, 
        #early_stopping_rounds=100,
        #,eval_set=[x_test, y_test]
        #eval_set=[(x_test, y_test)]
        ) 

# Evaluate model
model.fit(x_test, y_test)

y_predict = model.predict(x_test)
results = model.score(x_test, y_test)

acc = accuracy_score(y_test, y_predict)
print("최종점수:", results)
print("acc 는", acc)

print('============================')
count_data = train_csv.groupby('quality')['quality'].count
import matplotlib.pyplot as plt
plt.bar(count_data.index, count_data)
# plt.bar(y)
plt.show()


# submit
# save_path = './_data/'
# y_submit = model.predict(test_csv)
# submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)
# submission['quality'] = y_submit
# submission.to_csv(save_path + 'dacon_kaggle_submit.csv')