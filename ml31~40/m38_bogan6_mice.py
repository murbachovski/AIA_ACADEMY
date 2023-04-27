# Multiple Imputation by Chained Equations
import numpy as np
import pandas as pd
data = pd.DataFrame(
                    [[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
).transpose()
# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']

# print(data)
#      x1   x2    x3   x4
# 0   2.0  2.0   2.0  NaN
# 1   NaN  4.0   4.0  4.0
# 2   6.0  NaN   6.0  NaN
# 3   8.0  8.0   8.0  8.0
# 4  10.0  NaN  10.0  NaN

# 결측치 확인

# print(data.isnull()) # True가 결측치야
#       x1     x2     x3     x4
# 0  False  False  False   True
# 1   True  False  False  False
# 2  False   True  False   True
# 3  False  False  False  False
# 4  False   True  False   True

# print(data.isnull().sum())
# x1    1
# x2    2
# x3    0
# x4    3

# print(data.info())
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   x1      4 non-null      float64
#  1   x2      3 non-null      float64
#  2   x3      5 non-null      float64
#  3   x4      2 non-null      float64

# print('============결측치 삭제================')
# # print(data['x1'].dropna())   # 그 열에서만 삭제돼서 그게 그거다.
# # print(data.dropna())         # 행 위주로 삭제
# print('============================')
# # print(data.dropna(axis=0))     # 0 행 1 열# 위와 같다.
# print(data.dropna(axis=1))       # 0 행 1 열# 열 위주로 삭제한다.

# 2-1 특정 값 - 평균
# print('============결측치 처리 mean()=========')
# means = data.mean()
# print('평균: ', means)
# 평균  
# x1    6.500000
# x2    4.666667
# x3    6.000000
# x4    6.000000
# data2 = data.fillna(means)
# print(data2)

# 2-2 특정 값 - 중위값
# print('============결측치 처리 median()=========')
# median = data.median()
# data3 = data.fillna(median)
# print('중위값: ', median)
# 중위값:  x1    7.0
# x2    4.0
# x3    6.0
# x4    6.0
# dtype: float64
# print(data3)

# 2-3 특정 값 - ffill, bfill
# print('============결측치 처리 ffill, bfill=========')
# data4 = data.fillna(method='ffill')
# print(data4)
# data5 = data.fillna(method='backfill')
# print(data5)

# 2-4. 특정 값 - 임의 값으로 채우기
# print('============결측치 처리 임의의 값으로 채우기=========')
# data6 = data.fillna(77)
# data6 = data.fillna(value=777) # 위와 동일하다.
# print(data6)

#######################특정 칼럼만 !!!!!!!! ######################

# print(data)

# 1. x1컬럼에 평균값을 넣고
# means = data['x1'].mean()
# data['x1'] = data['x1'].fillna(means)
# print(data)

# 2. x2컬럼에 중위값을 넣고
# midean = data['x2'].median()
# data['x2'] = data['x2'].fillna(midean)
# print(data)

# 3. x4컬럼에 ffill한후 / 제일 위에 남은 행에 777777로 채우기
# data['x4'] = data['x4'].fillna(method='ffill')
# data['x4'] = data['x4'].fillna(value=77)

print(data)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
# imputer = SimpleImputer()                         # 디폴트는 평균!!!!!!!! 
# imputer = SimpleImputer(strategy='mean')          # 평균 
# imputer = SimpleImputer(strategy='median')        # 중위 
# imputer = SimpleImputer(strategy='most_frequent') # 최빈값 // 갯수가 같을 경우 가장 작은 값  
# imputer = SimpleImputer(strategy='constant', fill_value=777)        # 0 들어간다.         
# imputer = SimpleImputer(strategy='constant', fill_value=777)        # 0 들어간다.         
# imputer = KNNImputer() 
imputer = IterativeImputer(estimator=XGBRegressor())

data2 = imputer.fit_transform(data)
print(data2)
# [[ 2.          2.          2.          6.        ]
#  [ 6.5         4.          4.          4.        ]
#  [ 6.          4.66666667  6.          6.        ]
#  [ 8.          8.          8.          8.        ]
#  [10.          4.66666667 10.          6.        ]]