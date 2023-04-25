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
data4 = data.fillna(method='ffill')
# print(data4)
data5 = data.fillna(method='backfill')
# print(data5)

# 2-4. 특정 값 - 임의 값으로 채우기
# print('============결측치 처리 임의의 값으로 채우기=========')
# data6 = data.fillna(77)
data6 = data.fillna(value=777) # 위와 동일하다.
print(data6)



