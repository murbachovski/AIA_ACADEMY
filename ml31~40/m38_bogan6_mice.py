# Multiple Imputation by Chained Equations
import numpy as np
import pandas as pd
from impyute.imputation.cs import mice
data = pd.DataFrame(
                    [[2, np.nan, 6, 8, 10],
                    [2, 4, np.nan, 8, np.nan],
                    [2, 4, 6, 8, 10],
                    [np.nan, 4, np.nan, 8, np.nan]]
).transpose()
# print(data)
data.columns = ['x1', 'x2', 'x3', 'x4']



impute_df = mice(data.to_nupy()) # mice에는 numpy로 넣어줘!!
print(impute_df)
# [[ 2.          2.          2.          1.97121734]
#  [ 4.05785099  4.          4.          4.        ]
#  [ 6.          5.54446855  6.          5.83709445]
#  [ 8.          8.          8.          8.        ]
#  [10.          9.08893709 10.          9.70297156]]
# .values 와 .to_numpy() 넘파이로 바꿔준다.
# 판다스 -> 넘파이
# 넘파이 -> 판다스
# 리스트 -> 넘파이
# 넘파이 -> 리스트
#각각 예제 1개씩 만들어서 제출
