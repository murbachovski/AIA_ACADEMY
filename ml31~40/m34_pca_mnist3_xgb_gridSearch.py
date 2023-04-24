from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier as xgb
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold,cross_val_score, StratifiedKFold
# n_component › 0.95
# xgboost, gridsearch 또는 Randomsearch를 쓸것

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
# print(x.shape, y.shape) # (60000, 28, 28) (60000,)
x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape) # (70000, 28, 28) (70000,)
# x_train = np.array(x_train)
x = x.reshape(70000, 28*28)
# x_test = np.array(x_test)
# x_test = x_test.reshape(60000, 28*28)

pca = PCA(n_components=10)
x = pca.fit_transform(x)

# 결과를 뛰어넘어랏!!!
xgb_parameters = [
    {"_estimators": [100, 200, 300],
     "learning_rate": [0.1, 0.3, 0.001, 0.01],
    "max_depth": [4, 5, 6]},
    {"_estimators": [90, 100, 110],
    "learning_rate": [0.1, 0.001, 0.01],
    "max _depth": [4,5,6],
    "colsample_bytree": [0.6, 0.9, 1]},
    {"_estimators": [90, 110],
    "learning rate": [0.1, 0.001, 0.5],
    "max _depth": [4,5,6],
    "colsample _bytree": [0.6, 0.9, 1]},
    {"colsample_bylevel": [0.6, 0.7, 0.9]}
]


n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=337)

RSCV_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'degree': [3, 4, 5]}, # 12
    {'C': [1, 10, 100], 'kernel': ['rbf, lienear'], 'gamma': [0.001, 0.0001]}, # 12
    {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.01, 0.001, 0.0001], 'degree': [3, 4]}, # 24
    {'C':[0.1, 1], 'gamma': [1,10]} # 4
]

#2. MODEl
model = RandomizedSearchCV(xgb(),
                    RSCV_parameters,
                     cv=5,
                     verbose=1,
                     refit=True,
                     n_jobs=-1
)
model.fit(x, y)

y_pred = model.predict(x)
acc = accuracy_score(y, y_pred)
print('acc:', acc)
# acc: 0.9863