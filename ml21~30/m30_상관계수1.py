import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

#1. DATA
dataset = load_iris()
# print(dataset.feature_names) # 판다스는 columns
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = dataset['data']
y = dataset.target

df = pd.DataFrame(x, columns=dataset.feature_names)
# print(df)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#        0    1    2    3
# 0    5.1  3.5  1.4  0.2
# 1    4.9  3.0  1.4  0.2
# 2    4.7  3.2  1.3  0.2
# 3    4.6  3.1  1.5  0.2
# 4    5.0  3.6  1.4  0.2
# ..   ...  ...  ...  ...
# 145  6.7  3.0  5.2  2.3
# 146  6.3  2.5  5.0  1.9
# 147  6.5  3.0  5.2  2.0
# 148  6.2  3.4  5.4  2.3
# 149  5.9  3.0  5.1  1.8

# [150 rows x 4 columns]

df['Target(Y)'] = y
# print(df)
#      sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
# 0                  5.1               3.5                1.4               0.2          0
# 1                  4.9               3.0                1.4               0.2          0
# 2                  4.7               3.2                1.3               0.2          0
# 3                  4.6               3.1                1.5               0.2          0
# 4                  5.0               3.6                1.4               0.2          0
# ..                 ...               ...                ...               ...        ...
# 145                6.7               3.0                5.2               2.3          2
# 146                6.3               2.5                5.0               1.9          2
# 147                6.5               3.0                5.2               2.0          2
# 148                6.2               3.4                5.4               2.3          2
# 149                5.9               3.0                5.1               1.8          2

# [150 rows x 5 columns]

print("=====================상관계수 히트 맵 짜잔=====================")
print(df.corr()) #상관
#                    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  Target(Y)
# sepal length (cm)           1.000000         -0.117570           0.871754          0.817941   0.782561
# sepal width (cm)           -0.117570          1.000000          -0.428440         -0.366126  -0.426658
# petal length (cm)           0.871754         -0.428440           1.000000          0.962865   0.949035
# petal width (cm)            0.817941         -0.366126           0.962865          1.000000   0.956547
# Target(Y)                   0.782561         -0.426658           0.949035          0.956547   1.000000

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(font_scale=1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()