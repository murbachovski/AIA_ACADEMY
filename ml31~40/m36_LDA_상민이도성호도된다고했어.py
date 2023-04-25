import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits, fetch_california_housing
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
# # 회귀에서 된다고 했다!!

# # 1. DATA
x, y = fetch_california_housing(return_X_y=True)
# y = np.round(y)
# print(x.shape) # (50000, 3072)
print(y)
print(np.unique(y, return_counts=True))

# lda = LinearDiscriminantAnalysis()
# x = lda.fit_transform(x, y)

# print(x.shape) # (20640, 5)
####회귀는 원래 안되지만 diabetes는 정수형이라서  LDA에서 y의 클래스로 인식한거야
#그래서 돌아간거야

# 성호는 캘리포니아에서 라운드처리했어
#그러다보니 그것도 정수형이라서 클래스로 인식된거야.

# 회귀데이터는 원칙적으로 에러인데
# 위처럼 돌리고 싶으면 돌려도 되
# 성능 보장못함 ㅋㅋ