import numpy as np
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline # 파이프라인 들어갑니다.
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

#1. DATA
x, y = load_digits(return_X_y=True)
# sklearn의 모든 모델은 2차원만 받습니다
# print(x.shape) # (1797, 64)
# print(np.unique(y, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64))

# pca = PCA(n_components=8) # 몇개의 컬럼으로 압축을 할 것이다.
# x = pca.fit_transform(x)
# print(x.shape) # (1797, 8)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    shuffle=True,
    random_state=337,
    train_size=0.8
)

#2. MODEL
# model = RandomForestClassifier()

model = make_pipeline(PCA(n_components=8), RobustScaler(), SVC())

#3. COMPILE
model.fit(x_train, y_train)

#4. PREDICT
result = model.score(x_test, y_test)
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('model: ', result, 'ACC: ', acc)
