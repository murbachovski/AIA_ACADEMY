import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier # 이상치에 탁월하다, 스케일러 불필요


from sklearn.metrics import accuracy_score
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

#1. DATA

datasets = load_iris()
# datasets =  load_breast_cancer(return_X_y=True)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=337,
    train_size=0.8
)

# MODEL
model = RandomForestClassifier()

#3. COMPILE
model.fit(x_train, y_train)

#4. PREDICT
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('model: ', model, '\n', 'result: ',  result, '\n', 'ACC: ', acc, '\n', 'model_feature: ', model.feature_importances_)
print('==================================')
# print(model, '', model.feature_importances_) # TREE계열에만 존재합니다. feature...

def plot_feature_importances(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel('Feature_Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)
    plt.title(model)

plot_feature_importances(model)
plt.show()