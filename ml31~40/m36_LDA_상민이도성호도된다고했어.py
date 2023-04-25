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
