import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes, load_digits
from tensorflow.keras.datasets import cifar100
import matplotlib.pyplot as plt
iris = load_iris()
x = iris.data[:, 2:]
y = iris.target

# SCATTER PLOT 
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlabel('petal_length')
plt.xlabel('petal_width')
plt.title('iris scatter plot')
plt.show()
# # 컬럼의 갯수가 클래스의 갯수보다 작을때
# # 디폴트로 돌아가나?

# # 1. DATA
# # x, y = load_iris(return_X_y=True)
# # x, y = load_breast_cancer(return_X_y=True)
# # x, y = load_digits(return_X_y=True)
# (x_train, y_train), (x_test, y_test) = cifar100.load_data()
# # print(x_train.shape) # (50000, 32, 32, 3)
# x_train = x_train.reshape(50000, 32*32*3)
# # print(x_train.shape) # (50000, 3072)

# pca = PCA(n_components=5)
# x_train = pca.fit_transform(x_train)

# # print(x.shape) # (150, 3)

# lda = LinearDiscriminantAnalysis()
# # n_components는 클래스의 갯수 빼기 하나 이하로 가능하다.
# x = lda.fit_transform(x_train, y_train)
# print(x.shape)
